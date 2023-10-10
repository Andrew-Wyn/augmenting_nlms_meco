import numpy as np
from collections import defaultdict
from anm.utils import LOGGER, create_scheduler, load_model_from_hf
from anm.gaze_training.trainer import GazeTrainer
from anm.gaze_training.utils import create_finetuning_optimizer
from sklearn.model_selection import KFold
from anm.gaze_dataloader.dataset import minmax_preprocessing


def cross_validation(cf, dataset, tokenizer, DEVICE, writer, full_finetuning, k_folds=10):
    """
        Perform a k-fold cross-validation

        Args:
            cf: configuration of the training script
            dataset: the entire dataset object
            tokenizer: the tokenizer object
            DEVICE: the device (cpu|gpu) over the experiments will be execute
            writer: the object to write the trainer results
            k_folds: the fold number of k-fold cross-validation
        
        Results:
            loss_tr_mean: 
            loss_ts_mean:
    """

    folds = KFold(n_splits=k_folds)

    splits = folds.split(np.zeros(dataset.num_rows))

    loss_tr_mean = defaultdict(lambda: 0)
    loss_ts_mean = defaultdict(lambda: 0)

    LOGGER.info(f"Started Cross-Validation, with K = {k_folds}")

    for k, (train_idx, test_idx) in enumerate(splits):
        # cicle over folds, for every fold create train_d, valid_d
        LOGGER.info(f"Processing fold number {k+1}/{k_folds}")

        LOGGER.info("Creating train and test dataloaders...")
        # create the dataloader
        # train_dl
        train_dl, test_dl = minmax_preprocessing(cf, dataset, tokenizer, (train_idx, test_idx))

        LOGGER.info("Load the model from HF...")
        # Model
        model = load_model_from_hf(cf.model_type, cf.model_name, cf.pretrained, full_finetuning)

        # optimizer
        optim = create_finetuning_optimizer(cf, model)

        # scheduler
        scheduler = create_scheduler(cf, optim, train_dl)

        LOGGER.info("Starting model training...")
        # trainer
        trainer = GazeTrainer(cf, model, train_dl, optim, scheduler, f"CV-Training-{k+1}/{k_folds}",
                                    DEVICE, writer=writer, test_dl=test_dl)
        trainer.train()

        for key, metric in trainer.tester.train_metrics.items():
            loss_tr_mean[key] += metric

        for key, metric in trainer.tester.test_metrics.items():
            loss_ts_mean[key] += metric

    for key in loss_tr_mean:
        loss_tr_mean[key] /= k_folds

    for key in loss_ts_mean:
        loss_ts_mean[key] /= k_folds

    return loss_tr_mean, loss_ts_mean
