import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

# from anm.modeling.multitask_camembert import CamembertForMultiTaskTokenClassification
from anm.modeling.multitask_roberta import RobertaForMultiTaskTokenClassification
from anm.gaze_dataloader.datacollator import DataCollatorForMultiTaskTokenClassification
from anm.gaze_training.trainer import GazeTrainer
from anm.gaze_dataloader.dataset import minmax_preprocessing
from anm.utils import Config, load_model_from_hf, create_scheduler
from transformers import AdamW
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from anm.gaze_dataloader.dataset import _create_senteces_from_data, create_tokenize_and_align_labels_map
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    AutoTokenizer,
    set_seed,
)

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a XLM-Roberta-base following config json passed')
    parser.add_argument('-c' ,'--config', dest='config_file', action='store',
                        help=f'Relative path of a .json file, that contain parameters for the fine-tune script')
    parser.add_argument('-o', '--output-dir', dest='output_dir', action='store',
                        help=f'Relative path of output directory')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store',
                        help=f'Relative path of dataset folder, containing the .csv file')

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
    
    tf_logs_dir = args.output_dir+"/tf_logs"

    if not os.path.exists(tf_logs_dir):
        os.makedirs(tf_logs_dir)

    # Writer
    writer = SummaryWriter(tf_logs_dir)

    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, add_prefix_space=True)

    data = pd.read_csv("augmenting_nlms_meco_data/en/en_6_dataset.csv", index_col=0)
    gaze_dataset = _create_senteces_from_data(data)

    dataloader = minmax_preprocessing(cf, gaze_dataset, tokenizer)

    model = load_model_from_hf(cf.model_name, cf.pretrained)

    # optimizer
    optim = AdamW(model.parameters(), lr=cf.lr, eps=cf.eps)

    # scheduler
    scheduler = create_scheduler(cf, optim, dataloader)

    # trainer
    trainer = GazeTrainer(cf, model, dataloader, optim, scheduler, f"Final-retraining",
                                DEVICE, writer=writer, test_dl=None)
    trainer.train()


if __name__ == "__main__":
    main()