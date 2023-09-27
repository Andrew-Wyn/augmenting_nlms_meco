import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

import argparse

import torch

import numpy as np
import pandas as pd
from transformers import DataCollatorWithPadding

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # DataCollatorWithPadding,
    EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from anm.utils import Config, LOGGER
from datasets import Dataset


def read_complexity_dataset(path=None):
    data = []

    df = pd.read_csv(path)

    for _, row in df.iterrows():
        
        num_individuals = 20

        text = row["SENTENCE"]

        label = 0

        for i in range(num_individuals):
            label += int(row[f"judgement{i+1}"])

        label = label/num_individuals

        data.append({
            "text": text,
            "label": label
        })


    return Dataset.from_list(data)


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_hf(model_name, pretrained):

    # Model
    LOGGER.info("Initiating model ...")
    if not pretrained:
        # initiate model with random weights
        LOGGER.info("Take randomized model")
        
        config = AutoConfig.from_pretrained(model_name,
                                            num_labels=1,
                                            output_attentions=False,
                                            output_hidden_states=False)
        
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        LOGGER.info("Take pretrained model")
    
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=1,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)

    return model


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    parser.add_argument('-dd', '--dev_dataset', dest='dev_dataset', action='store',
                        help=f'Relative path of development dataset folder, containing the .csv file')
    parser.add_argument('-td', '--test_dataset', dest='test_dataset', action='store',
                        help=f'Relative path of test dataset folder, containing the .csv file')
    parser.add_argument('-m', '--model-dir', dest='model_dir', action='store',
                        help=f'Relative path of finetuned model directory, containing the config and the saved weights')
    parser.add_argument('-n', '--model-name', dest='model_name', action='store',
                        help=f'name of the model that you want to use')
    parser.add_argument('-p', '--pretrained', dest='pretrained', default=False, action='store_true',
                        help=f'Bool, start from a pretrained model')
    parser.add_argument('-f', '--finetuned', dest='finetuned', default=False, action='store_true',
                        help=f'Bool, start from a finetuned model')

    # Read the script's argumenents
    args = parser.parse_args()
    config_file = args.config_file

    # Load the .json configuration file
    cf = Config.load_json(config_file)

    # set seed
    set_seed(cf.seed)

    # check if the output directory exists, if not create it!
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # development dataset loading
    train_dataset = read_complexity_dataset(args.dev_dataset)

    # test dataset loading
    test_dataset = read_complexity_dataset(args.test_dataset)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)

    # Tokenize datasets
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_train_ds = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_ds = test_dataset.map(preprocess_function, batched=True)

    """
    Now create a batch of examples using DataCollatorWithPadding.
    It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation,
    instead of padding the whole dataset to the maximum length.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        num_train_epochs=cf.n_epochs,              # total number of training epochs
        per_device_train_batch_size=cf.train_bs,  # batch size per device during training
#        per_device_eval_batch_size=cf.eval_bs,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=cf.weight_decay,               # strength of weight decay
#        save_strategy="epoch",
#        evaluation_strategy="epoch",
        learning_rate=cf.lr,
#        load_best_model_at_end = True,  
#        metric_for_best_model = 'rmse',
#        save_total_limit = 1, # Only last model are saved. Older ones are deleted.
#        greater_is_better = False
    )

    """
    /see: https://towardsdatascience.com/linear-regression-with-hugging-face-3883fe729324

    According to Hugging Face’s library, when we load the pre-trained models from the Hugging Face API, 
    setting the num_labels to 1 for the AutoModelForSequenceClassification will trigger the linear regression and use 
    MSELoss() as the loss function automatically.
    """
    # Model
    if not args.finetuned: # downaload from huggingface
        LOGGER.info("Model retrieving, not finetuned, from hf...")
        model = load_model_from_hf(args.model_name, args.pretrained)
    else: # the finetuned model has to be loaded from disk
        LOGGER.info("Model retrieving, finetuned, load from disk...")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, 
                                                                   num_labels=1, 
                                                                   ignore_mismatched_sizes=True, 
                                                                   output_attentions=False, output_hidden_states=False)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_train_ds,         # training dataset
#        eval_dataset=tokenized_test_ds,            # evaluation dataset
        compute_metrics=compute_metrics_for_regression,
        tokenizer = tokenizer,
        data_collator=data_collator,
#        callbacks = [EarlyStoppingCallback(early_stopping_patience=cf.patience)]
    )

    train_result = trainer.train()

    trainer.save_model()

    # compute train results
    # metrics = train_result.metrics


    # compute evaluation results
    train_metrics = trainer.evaluate(eval_dataset=tokenized_train_ds, metric_key_prefix="train")
    # save train results
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)


    # compute evaluation results
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test_ds, metric_key_prefix="test")

    # save evaluation results
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)