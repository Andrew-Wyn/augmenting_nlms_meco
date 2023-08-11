import numpy as np
from collections import defaultdict
from anm.utils import LOGGER, create_finetuning_optimizer, create_scheduler, minMaxScaling, load_model_from_hf
from anm.gaze_dataloader.dataloader import GazeDataLoader
from anm.gaze_training.trainer import GazeTrainer


def cross_validation(cf, d, writer, DEVICE, k_folds=10):
    """
    Perform a k-fold cross-validation
    """

    l = len(d.text_inputs)
    l_ts = l//k_folds

    loss_tr_mean = defaultdict(int)
    loss_ts_mean = defaultdict(int)

    for k in range(k_folds):
        # cicle over folds, for every fold create train_d, valid_d
        if k != k_folds-1: # exclude the k-th part from the validation
            train_inputs = np.append(d.text_inputs[:(k)*l_ts], d.text_inputs[(k+1)*l_ts:], axis=0)
            train_targets = np.append(d.targets[:(k)*l_ts], d.targets[(k+1)*l_ts:], axis=0)
            train_masks = np.append(d.masks[:(k)*l_ts], d.masks[(k+1)*l_ts:], axis=0)
            test_inputs = d.text_inputs[k*l_ts:(k+1)*l_ts]
            test_targets = d.targets[k*l_ts:(k+1)*l_ts]
            test_masks = d.masks[k*l_ts:(k+1)*l_ts]

        else: # last fold clausole
            train_inputs = d.text_inputs[:k*l_ts]
            train_targets = d.targets[:k*l_ts]
            train_masks = d.masks[:k*l_ts]
            test_inputs = d.text_inputs[k*l_ts:]
            test_targets = d.targets[k*l_ts:]
            test_masks = d.masks[k*l_ts:]

        LOGGER.info(f"Train data: {len(train_inputs)}")
        LOGGER.info(f"Test data: {len(test_inputs)}")

        # min max scaler the targets
        train_targets, test_targets = minMaxScaling(train_targets, test_targets, d.feature_max)

        # create the dataloader
        train_dl = GazeDataLoader(cf, train_inputs, train_targets, train_masks, d.target_pad, mode="train")
        test_dl = GazeDataLoader(cf, test_inputs, test_targets, test_masks, d.target_pad, mode="test")

        # Model
        model = load_model_from_hf(cf.model_name, not cf.random_weights, cf.multiregressor, d.d_out)

        # optimizer
        optim = create_finetuning_optimizer(cf, model)

        # scheduler
        scheduler = create_scheduler(cf, optim, train_dl)

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
