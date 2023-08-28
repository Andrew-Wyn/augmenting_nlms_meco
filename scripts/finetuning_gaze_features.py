import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

# from anm.modeling.multitask_camembert import CamembertForMultiTaskTokenClassification
from anm.gaze_training.trainer import GazeTrainer
from anm.gaze_dataloader.dataset import minmax_preprocessing
from anm.gaze_training.cv import cross_validation
from anm.utils import Config, load_model_from_hf, create_scheduler, LOGGER
from transformers import AdamW
import pandas as pd
from anm.gaze_dataloader.dataset import _create_senteces_from_data
import argparse
import torch
import json
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

    data = pd.read_csv(args.dataset, index_col=0)
    gaze_dataset = _create_senteces_from_data(data)

    # --- 10-Fold Cross Validation
    loss_tr_mean, loss_ts_mean = cross_validation(cf, gaze_dataset, tokenizer, DEVICE, writer, k_folds=2)

    # --- Final Retraining

    dataloader = minmax_preprocessing(cf, gaze_dataset, tokenizer)

    model = load_model_from_hf(cf.model_type, cf.model_name, cf.pretrained)

    # optimizer
    optim = AdamW(model.parameters(), lr=cf.lr, eps=cf.eps, weight_decay=cf.weight_decay)

    # scheduler
    scheduler = create_scheduler(cf, optim, dataloader)

    # trainer
    trainer = GazeTrainer(cf, model, dataloader, optim, scheduler, f"Final-retraining",
                                DEVICE, writer=writer, test_dl=None)
    trainer.train()

    LOGGER.info ("Saving metrics...")
    # save cv and final train metrics
    with open(f"{args.output_dir}/finetuning_results.json", 'w') as f:
        json.dump({"losses_tr" : loss_tr_mean, "losses_ts" : loss_ts_mean, "final_training" : trainer.tester.train_metrics.items()}, f)


if __name__ == "__main__":
    main()