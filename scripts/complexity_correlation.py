import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

import argparse

import torch
from tqdm import tqdm

from scipy import stats

import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

from datasets import Dataset

import json


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR



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


def main(args):

    # check if the output directory exists, if not create it!
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # development dataset loading
    train_dataset = read_complexity_dataset(args.dev_dataset)

    # test dataset loading
    test_dataset = read_complexity_dataset(args.test_dataset)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, 
                                                                ignore_mismatched_sizes=True, 
                                                                output_attentions=False, output_hidden_states=False)

    # Correlation

    # Train datataset
    # Test dataset
    values = {
        "train_predicted": [],
        "train_labels": [],
        "train_corr_s": [],
        "train_corr_p": [],
        "test_predicted": [],
        "test_labels": [],
        "test_corr_s": [],
        "test_corr_p": []
    }

    with torch.no_grad():

        for sample in tqdm(train_dataset):
            inputs = tokenizer(sample["text"], truncation=True, return_tensors="pt")
            logits = model(**inputs).logits

            values["train_predicted"].append(float(logits.numpy()[0][0]))
            values["train_labels"].append(sample["label"])

        # compute correlation
        values["train_corr_s"] = float(stats.spearmanr(values["train_predicted"], values["train_labels"]).statistic)
        values["train_corr_p"] = float(stats.pearsonr(values["train_predicted"], values["train_labels"]).statistic)

        for sample in tqdm(test_dataset):
            inputs = tokenizer(sample["text"], truncation=True, return_tensors="pt")
            logits = model(**inputs).logits

            values["test_predicted"].append(float(logits.numpy()[0][0]))
            values["test_labels"].append(sample["label"])

        # compute correlation
        values["test_corr"] = float(stats.spearmanr(values["test_predicted"], values["test_labels"]).statistic)
        values["test_corr"] = float(stats.pearsonr(values["test_predicted"], values["test_labels"]).statistic)


    with open(args.output_dir+"/"+'ouput.json', 'w') as f:
        json.dump(values, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Correlation between complexity tuned model predictions and the dataset labels')
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
    
    # Read the script's argumenents
    args = parser.parse_args()

    # set seed
    set_seed(123)

    main(args)