import numpy as np
from collections import defaultdict
from anm.utils import LOGGER, create_scheduler, load_model_from_hf
from anm.gaze_training.trainer import GazeTrainer
from sklearn.model_selection import KFold
from transformers import AdamW
from anm.gaze_dataloader.dataset import minmax_preprocessing


def cross_validation(cf, dataset, tokenizer, DEVICE, writer, k_folds=10):
    """
    Perform a k-fold cross-validation
    """

    folds = KFold(n_splits=k_folds)

    splits = folds.split(np.zeros(dataset.num_rows))

    loss_tr_mean = defaultdict(int)
    loss_ts_mean = defaultdict(int)

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
        model = load_model_from_hf(cf.model_name, cf.pretrained)

        # optimizer
        optim = AdamW(model.parameters(), lr=cf.lr, eps=cf.eps)

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
