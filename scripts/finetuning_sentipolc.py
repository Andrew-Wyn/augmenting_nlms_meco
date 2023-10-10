import sys
import os

sys.path.append(os.path.abspath("."))

from anm.modeling.multitask_xlm_roberta import XLMRobertaForMultiTaskSequenceClassification
from anm.modeling.multitask_camembert import CamembertForMultiTaskSequenceClassification
from transformers import AutoTokenizer, AutoConfig, get_scheduler
from anm.gaze_dataloader.datacollator import MultiLabelDataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import evaluate
import argparse
import torch
import json
import math
import csv

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def create_dataset_from_faulty_csv(src_path):
    dataset_dict = {'text': [], 'label_pos': [], 'label_neg': []}
    with open(src_path) as src_file:
        csv_reader = csv.reader(src_file, delimiter=',', quotechar='"')
        print('')
        for row in csv_reader:
            if row[0] == 'idtwitter':
                continue
            if len(row) != 9:
                cut_row = row[:9]
                cut_row[8] += ',' + ', '.join(row[9:])
                row = cut_row
            dataset_dict['text'].append(row[8])
            dataset_dict['label_pos'].append(int(row[2]))
            dataset_dict['label_neg'].append(int(row[3]))
    return Dataset.from_dict(dataset_dict)


def prepare_datasets(dataset_dir, tokenizer):
    sentipolc_files = {
        'train': [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir) if
                  'training_set' in file_name][0],
        'test':
            [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir) if 'test_set' in file_name][
                0]
    }
    train_dataset = create_dataset_from_faulty_csv(sentipolc_files['train'])
    test_dataset = create_dataset_from_faulty_csv(sentipolc_files['test'])

    def preprocess_function(examples):
        result = tokenizer(examples["text"], truncation=True)
        return result

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text'])
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

    return tokenized_train_dataset, tokenized_test_dataset


def get_finetuned_model_path(model_cf, finetuned_models_dir, user_id):
    model_str = 'xlm' if model_cf.language_mode == 'cross_lingual' else 'camem'
    finetuned_str = 'p' if model_cf.finetuned else 'np'
    model_dir = f'{finetuned_models_dir}/gaze_finetuning_it_{user_id}_{finetuned_str}_{model_str}'
    for file_name in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file_name)
        if file_name != 'tf_logs' and os.path.isdir(file_path):
            if 'config.json' in os.listdir(file_path):
                model_path = file_path
            else:
                inner_dir = os.listdir(file_path)[0]
                model_path = os.path.join(file_path, inner_dir)
    return model_path


def get_config_with_tasks(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.tasks = ['pos', 'neg']
    return config


def load_model(model_cf, finetuned_models_dir=None, user_id=None):
    if model_cf.finetuned:
        model_name = get_finetuned_model_path(model_cf, finetuned_models_dir, user_id)
        print(f'Model name = {model_name}')

        config = get_config_with_tasks(model_name)
        if model_cf.language_mode == 'cross_lingual':
            model = XLMRobertaForMultiTaskSequenceClassification.from_pretrained(model_name, config=config,
                                                                                 ignore_mismatched_sizes=True)
        else:
            model = CamembertForMultiTaskSequenceClassification.from_pretrained(model_name, config=config,
                                                                                ignore_mismatched_sizes=True)
    else:
        if model_cf.language_mode == 'cross_lingual':
            model_name = 'xlm-roberta-base'
            print(f'Model name = {model_name}')

            config = get_config_with_tasks(model_name)
            if model_cf.pretrained:
                model = XLMRobertaForMultiTaskSequenceClassification.from_pretrained(model_name, config=config)
            else:
                model = XLMRobertaForMultiTaskSequenceClassification(config=config)
        else:
            model_name = 'idb-ita/gilberto-uncased-from-camembert'
            print(f'Model name = {model_name}')

            config = get_config_with_tasks(model_name)
            if model_cf.pretrained:
                model = CamembertForMultiTaskSequenceClassification.from_pretrained(model_name, config=config)
            else:
                model = CamembertForMultiTaskSequenceClassification(config=config)
    return model, model_name


def evaluate_model(model, dataloader, split, all_metrics, pos_metrics, neg_metrics):
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": {k: v.to(device) for k, v in batch["labels"].items()}
            }

            model_output = model(**batch)

            for task in batch['labels']:
                predictions = model_output.logits[task].argmax(dim=-1)
                references = batch['labels'][task]
                # for metric in metrics:
                all_metrics.add_batch(predictions=predictions, references=references)
                if task == 'pos':
                    pos_metrics.add_batch(predictions=predictions, references=references)
                elif task == 'neg':
                    neg_metrics.add_batch(predictions=predictions, references=references)

    all_res = all_metrics.compute()
    pos_res = pos_metrics.compute()
    neg_res = neg_metrics.compute()

    res_dict = {}

    for metric in all_res:
        res_dict[f'{split}_{metric}'] = all_res[metric]
    for metric in pos_res:
        res_dict[f'{split}_pos_{metric}'] = pos_res[metric]
    for metric in neg_res:
        res_dict[f'{split}_neg_{metric}'] = neg_res[metric]

    return res_dict


def get_eval_metrics():
    return evaluate.combine([
        evaluate.load("accuracy"),
        evaluate.load("f1"),
        evaluate.load("precision"),
        evaluate.load("recall")])


def get_out_path(out_dir, model_cf):
    model_str = 'xlm' if model_cf.language_mode == 'cross_lingual' else 'camem'
    pretrained = 'p' if model_cf.pretrained else 'np'
    finetuned = f'f_it{model_cf.user_id}' if model_cf.finetuned else 'nf'
    if not model_cf.finetuned and model_cf.language_mode == 'cross_lingual':
        model_str += '_it'
    return os.path.join(out_dir, f'{model_str}_{pretrained}_{finetuned}')


class TrainingConfig:

    def __init__(self, args):
        self.weight_decay = 1e-2
        self.lr = 2e-5 #5e-6
        self.train_bs = 8
        self.eval_bs = 8
        self.n_epochs = 8
        self.seed = 1234
        self.num_warmup_steps = 0
        self.language_mode = args.language_mode
        self.pretrained = args.pretrained
        self.finetuned = args.finetuned
        self.user_id = args.user_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained', dest='pretrained', default=False, action='store_true',
                        help=f'Bool, start from a pretrained model')
    parser.add_argument('-f', '--finetuned', dest='finetuned', default=False, action='store_true',
                        help=f'Bool, start from a finetuned model')
    parser.add_argument('-l', '--language_mode', dest='language_mode', choices=['cross_lingual', 'mono_lingual'])
    parser.add_argument('-u', '--user_id', dest='user_id', default=None, type=int, choices=[1, 26, 38, 43, 44])

    # Read the script's argumenents
    args = parser.parse_args()

    cf = TrainingConfig(args)

    print(f'Pretrained = {cf.pretrained}')
    print(f'Finetuned = {cf.finetuned}')
    print(f'Language mode = {cf.language_mode}')


    dataset_dir = 'augmenting_nlms_meco_data/sentiment/it_sentipolc'
    finetuned_models_dir = '/home/lmoroni/__workdir/augmenting_nlms_meco/output'
    model_save_dir = 'output/sentipolc'
    out_path = get_out_path(model_save_dir, cf)

    if os.path.exists(out_path):
        return

    model, model_name = load_model(cf, finetuned_models_dir=finetuned_models_dir, user_id=cf.user_id)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer_name = 'idb-ita/gilberto-uncased-from-camembert' if cf.language_mode == 'mono_lingual' else 'xlm-roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset, test_dataset = prepare_datasets(dataset_dir, tokenizer)
    data_collator = MultiLabelDataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                  batch_size=cf.train_bs)
    eval_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=cf.eval_bs)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cf.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cf.lr)#create_finetuning_optimizer(cf, model)
    num_training_steps = cf.n_epochs * math.ceil(len(train_dataloader))
    lr_scheduler = get_scheduler(name='linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=cf.num_warmup_steps,
                                 num_training_steps=num_training_steps)

    ## training loop

    progress_bar = tqdm(range(num_training_steps))
    model.to(device)

    f1 = evaluate.load('f1')

    for epoch in range(1, cf.n_epochs + 1):
        print('Epoch', epoch)
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": {k: v.to(device) for k, v in batch["labels"].items()}
            }

            model_output = model(**batch)
            loss = model_output.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cf.max_grad_norm)

        print("Loss", loss.item())

        model.eval()

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": {k: v.to(device) for k, v in batch["labels"].items()}
                }

                model_output = model(**batch)

                for task in batch['labels']:
                    predictions = model_output.logits[task].argmax(dim=-1)
                    references = batch['labels'][task]
                    f1.add_batch(predictions=predictions, references=references)

        eval_accuracy = f1.compute()
        print(f'Eval f1 = {eval_accuracy["f1"]}')

    model.save_pretrained(out_path)

    train_res = evaluate_model(model, train_dataloader, 'train', get_eval_metrics(), get_eval_metrics(),
                               get_eval_metrics())
    test_res = evaluate_model(model, eval_dataloader, 'test', get_eval_metrics(), get_eval_metrics(),
                              get_eval_metrics())

    metrics_out_path = os.path.join(out_path, 'all_results.json')
    with open(metrics_out_path, 'w+') as out_file:
        json.dump(train_res | test_res, out_file)


if __name__ == '__main__':
    main()
