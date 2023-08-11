import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

from anm.gaze_dataloader.dataloader import GazeDataLoader
from anm.gaze_dataloader.dataset import GazeDataset
from anm.gaze_training.cv import cross_validation
from anm.gaze_training.trainer import GazeTrainer
import torch
import json
from anm.utils import create_finetuning_optimizer, create_scheduler, Config, minMaxScaling, load_model_from_hf
from transformers import (
    AutoTokenizer,
    set_seed,
)

from torch.utils.tensorboard import SummaryWriter
import argparse


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

    # Writer
    writer = SummaryWriter(args.output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cf.model_name, cache_dir=CACHE_DIR)

    # Dataset
    d = GazeDataset(cf, tokenizer, args.dataset)
    d.read_pipeline()
    d.randomize_data()

    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(d.text_inputs[0])))
    print(d.targets)

    exit()

    # K-fold cross-validation
    train_losses, test_losses = cross_validation(cf, d, writer, DEVICE, k_folds=cf.k_folds)

    print("Train averaged losses:")
    print(train_losses)

    print("Test averaged losses:")
    print(test_losses)

    # Retrain over all dataset

    # min max scaler the targets
    d.targets = minMaxScaling(d.targets, feature_max=d.feature_max, pad_token=d.target_pad)

    # create the dataloader
    train_dl = GazeDataLoader(cf, d.text_inputs, d.targets, d.masks, d.target_pad, mode="train")

    # Model
    model = load_model_from_hf(cf.model_name, not cf.random_weights, cf.multiregressor, d.d_out)

    # Optimizer
    optim = create_finetuning_optimizer(cf, model)

    # Scheduler
    scheduler = create_scheduler(cf, optim, train_dl)

    # Trainer
    trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, f"Final_Training",
                                DEVICE, writer=writer)
    trainer.train(save_model=True, output_dir=args.output_dir)

    loss_tr = dict()

    for key, metric in trainer.tester.train_metrics.items():
        loss_tr[key] = metric

    with open(f"{args.output_dir}/finetuning_results.json", 'w') as f:
        json.dump({"losses_tr" : train_losses, "losses_ts" : test_losses, "final_training" : loss_tr}, f)

if __name__ == "__main__":
    main()