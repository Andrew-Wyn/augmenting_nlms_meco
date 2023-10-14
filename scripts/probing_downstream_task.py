import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

import argparse

import pandas as pd

from datasets import Dataset

from anm.utils import LOGGER, Config, load_model_from_hf
from anm.gaze_probing.downstream_prober import DownstreamProber
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    set_seed,
)

from transformers import DataCollatorWithPadding


from torch.utils.data import DataLoader


def load_model_from_hf(model_name, pretrained):
    # Model
    LOGGER.info("Initiating model ...")
    if not pretrained:
        # initiate model with random weights
        LOGGER.info("Take randomized model")
        
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        LOGGER.info("Take pretrained model")
    
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model


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
            "label": 1 if label >= 3.5 else 0
        })


    return Dataset.from_list(data)


# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

def main():
    parser = argparse.ArgumentParser(description='Regression Probing')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    parser.add_argument('-d', '--dataset-train', dest='train_dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')
    parser.add_argument('-t', '--dataset-test', dest='test_dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')
    parser.add_argument('-m', '--model-name', dest='model_name', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')



    # Load the script's arguments
    args = parser.parse_args()

    config_file = args.config_file
    output_dir = args.output_dir

    # check if the output directory exists, if not create it!
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load config file
    cf = Config.load_json(config_file)

    # set seed
    set_seed(cf.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # DataLoader
    # development dataset loading
    train_dataset = read_complexity_dataset(args.train_dataset)

    # test dataset loading
    test_dataset = read_complexity_dataset(args.test_dataset)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=CACHE_DIR)

    # Tokenize datasets
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_train_ds = train_dataset.map(preprocess_function, batched=True,
                                           remove_columns=["text"])
    tokenized_test_ds = test_dataset.map(preprocess_function, batched=True,
                                         remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer)
    dl_train = DataLoader(tokenized_train_ds, collate_fn=data_collator, batch_size=1)
    dl_test = DataLoader(tokenized_test_ds, collate_fn=data_collator, batch_size=1)

    # Model
    LOGGER.info("Model retrieving, from hf...")
    model = load_model_from_hf(cf.model_name, cf.pretrained)

    prober = DownstreamProber(dl_train, dl_test, output_dir)

    _ = prober.create_probing_dataset(model, tokenizer)

    prober.probe(cf.linear, cf.k_folds)


if __name__ == "__main__":
    main()