import os
import argparse
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

import torch
import pandas as pd

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    set_seed
)

from datasets import Dataset, load_dataset

from anm.modeling.multitask_roberta import RobertaForMultiTaskTokenClassification
from anm.modeling.multitask_xlm_roberta import XLMRobertaForMultiTaskTokenClassification
from anm.modeling.multitask_camembert import CamembertForMultiTaskTokenClassification
from anm.utils import Config


# --- Set HF CACHE and DEVICE
# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAPPING = {
    "roberta": "roberta-base",
    "xlm": "xlm-roberta-base",
    "camem": "idb-ita/gilberto-uncased-from-camembert"
}

DATASET_PATH = {
    "complexity_en": "augmenting_nlms_meco_data/complexity_data/complexity_ds_en_test.csv",
    "complexity_it": "augmenting_nlms_meco_data/complexity_data/complexity_ds_it_test.csv",
    "sentiment_en": "sst2",
    # "sentiment_it": "..." # TODO: LUCADINI, aggiungere path del dataset di sentipolc
}

# Dataset Utilities

def read_complexity_dataset(path=None):
    data = []

    df = pd.read_csv(path)

    for _, row in df.iterrows():
        
        text = row["SENTENCE"]

        data.append({
            "sentence": text
        })


    return Dataset.from_list(data)


def read_sentiment_it_dataset(path_None):
    # TODO: LUCADINI, aggiungere dataset loading di sentipolc, the text field sould be named "sentence", following the sst2 notation
    pass

# ---

def create_dataset(task):

    dataset_path = DATASET_PATH[task]

    # get out the dataset from different paths, only the test-set
    if task == "complexity_en":
        dataset = read_complexity_dataset(dataset_path)
    elif task == "complexity_it":
        dataset = read_complexity_dataset(dataset_path)
    elif task == "sentiment_en":
        dataset = load_dataset(dataset_path, cache_dir=CACHE_DIR)
    elif task == "sentiment_it":
        dataset = read_sentiment_it_dataset(dataset_path)

    return dataset


def update_dataset(annotaded_dataset, splitted_text, logits, word_id, ianum, sentence_num):

    # align subwords and logits

    aligned_logits = {k: [] for k in logits.keys()}

    actual_w = None
    for i, w in enumerate(word_id):
        if w == None:
            continue
        
        if w==actual_w:
            continue

        for t in aligned_logits.keys():
            aligned_logits[t].append(logits[t].flatten().tolist()[i])

        actual_w = w

    # update the annotated_dataset structure

    for i, ia in enumerate(splitted_text):
        annotaded_dataset["trialid"].append(-1)
        annotaded_dataset["sentnum"].append(sentence_num)
        annotaded_dataset["ianum"].append(ianum)
        annotaded_dataset["ia"].append(ia)

        for t in aligned_logits.keys():
            annotaded_dataset[t].append(aligned_logits[t][i])

        ianum += 1

    return ianum

def annotate_dataset(model, tokenizer, dataset, args):

    annotated_dataset = {
        "trialid": [],
        "sentnum": [],
        "ianum": [],
        "ia": [],
        "skip": [],
        "firstfix_dur": [],
        "firstrun_dur": [],
        "dur": [],
        "firstrun_nfix": [],
        "nfix": [],
        "refix": [],
        "reread": []
    }
    
    ianum = 0

    with torch.no_grad():

        for sentence_num, sample in tqdm(enumerate(dataset, 1)):
            splitted_text = sample["sentence"].split(" ")
            tokenized_input = tokenizer(splitted_text, 
                                        max_length=128, 
                                        padding=True, 
                                        truncation=True, 
                                        is_split_into_words=True, 
                                        return_tensors="pt")
            
            word_ids = tokenized_input.word_ids(batch_index=0)

            logits = model(**tokenized_input).logits

            ianum = update_dataset(annotated_dataset, splitted_text, logits, word_ids, ianum, sentence_num)

            if sentence_num == int(args.sentence_number):
                break

    return annotated_dataset


def save_dataset(annotated_dataset, args):
    pd_ds = pd.DataFrame.from_dict(annotated_dataset)

    pd_ds.to_csv(args.output_path)

def main(args):

    # Load Dataset
    dataset = create_dataset(args.task)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[args.model_name], add_prefix_space=True)
    
    # Model
    cf = Config.load_json("configs/modeling_configuration.json")
    config = AutoConfig.from_pretrained(args.model_dir)
    config.update({"tasks": cf.tasks})

    if args.model_name == "roberta":
        model = RobertaForMultiTaskTokenClassification.from_pretrained(args.model_dir, config=config)
    elif args.model_name == "xlm":
        model = XLMRobertaForMultiTaskTokenClassification.from_pretrained(args.model_dir, config=config)
    elif args.model_name == "camem":
        model = CamembertForMultiTaskTokenClassification.from_pretrained(args.model_dir, config=config)

    annotated_dataset = annotate_dataset(model, tokenizer, dataset, args)

    save_dataset(annotated_dataset, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create silver labels dataset from gaze features')
    parser.add_argument('-m', '--model-dir', dest='model_dir', action='store',
                        help=f'Relative path of gaze tuned model')
    parser.add_argument('-n', '--model-name', dest='model_name', action='store',
                        help=f'Name of the model')
    parser.add_argument('-t', '--task', dest='task', action='store',
                        help=f'Task type, could be: complexity_en, complexity_it, sentiment_en, sentiment_it')
    parser.add_argument('-o', '--output-path', dest='output_path', action='store',
                        help=f'Relative path of output dataset')
    parser.add_argument('-s', '--sentence-number', dest='sentence_number', action='store',
                        help=f'Number of sentence to annotate')
    
    # Read the script's argumenents
    args = parser.parse_args()

    # set seed
    set_seed(123)

    main(args)