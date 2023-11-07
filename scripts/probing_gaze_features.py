import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

import argparse

import pandas as pd

import torch

from anm.gaze_dataloader.datacollator import DataCollatorForMultiTaskTokenClassification
from torch.utils.data import DataLoader

from anm.utils import LOGGER, Config, load_model_from_hf
from anm.gaze_probing.prober import Prober
from anm.gaze_dataloader.dataset import _create_senteces_from_data, create_tokenize_and_align_labels_map
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)


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
    parser.add_argument('-d', '--dataset', dest='dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')
    parser.add_argument('-p', '--pretrained', dest='pretrained', default=False, action='store_true',
                        help=f'Bool, start from a pretrained model')

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
    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, add_prefix_space=True)

    # DataLoader
    data = pd.read_csv(args.dataset, index_col=0)
    modeling_cf = Config.load_json("configs/modeling_configuration.json")
    gaze_dataset = _create_senteces_from_data(data, modeling_cf.tasks)

    features = [col_name for col_name in gaze_dataset.column_names if col_name.startswith('label_')]

    ## Align labels
    tokenized_dataset = gaze_dataset.map(
                create_tokenize_and_align_labels_map(tokenizer, features),
                batched=True,
                remove_columns=gaze_dataset.column_names,
    )
    
    ## DataCollator
    data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
    dataloader = DataLoader(tokenized_dataset, shuffle=True, collate_fn=data_collator, batch_size=cf.train_bs)

    # Model
    LOGGER.info("Model retrieving, from hf...")
    model = load_model_from_hf(cf.model_type, cf.model_name, args.pretrained, partial_finetuning=False)

    prober = Prober(dataloader, output_dir)

    _ = prober.create_probing_dataset(model)

    prober.probe(cf.linear, cf.k_folds)


if __name__ == "__main__":
    main()