import numpy as np
from anm.utils import LOGGER
from collections import defaultdict
from tqdm import tqdm
import torch
import datetime
import json

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from scipy import stats
from sklearn.model_selection import KFold

from tqdm import tqdm


def binary_accuracy(targets, predicted, thr):
    thresholded = np.array(predicted) 

    thresholded[thresholded >= thr] = 100
    thresholded[thresholded < thr] = 0

    return accuracy_score(targets, thresholded)

class Prober():
    """
        Class used to compute the probing experiments
    """

    def __init__(self, dataloader, output_dir):
        """
            Args:
                dataloader: the dataloader of the dataset
                output_dir (str): the path where the resutls will be saved
        """
        self.dataloader = dataloader
        self.probing_dataset = None
        self.output_dir = output_dir

    def create_probing_dataset(self, model):
        """
            Create probing dataset, given the model to probe.

            Args:
                model: the model over which compute the probing
            Returns:
                probing_dataset (dict): the probing dataset, with a structure defined by us.
                {"layers": {"i": np.array}, "labels": {"label_name": []}}
        """
        LOGGER.info(f"Creating probing datasets...")

        probing_dataset = defaultdict(list)

        probing_dataset = {
            "layers": {}, # for each layer it contains the hidden representations of each token (first sub-token)
            "labels": defaultdict(list) # for each label it contains the list of the target values
        }

        # cicle over the dataloader, one batch at time, assumed to have batch_size = 1
        for batch in tqdm(self.dataloader):
            batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": {k: v for k, v in batch["labels"].items()}
            }

            target = batch["labels"]
            
            non_masked_els = None
            
            # for each label fill the relative storage in the returning dict, taking out the values == -100
            for label, targets in target.items():
                    targets = targets[0]
                    non_masked_els = (targets != -100) > 0
                    probing_dataset["labels"][label] += targets[non_masked_els].tolist()
            
            # pass the input in the model to retrieve the hidden states
            with torch.no_grad():
                model_output = model(**batch, output_hidden_states=True)

            # for each layer fill the relative storage in the returning dict, taking only the first sub-token elements
            for layer in range(model.config.num_hidden_layers):

                hidden_state = model_output.hidden_states[layer].numpy()
                    
                probe_input = hidden_state[0, non_masked_els, :]

                if layer in probing_dataset["layers"].keys():
                    probing_dataset["layers"][layer] = np.concatenate([probing_dataset["layers"][layer], probe_input])
                else:
                    probing_dataset["layers"][layer] = probe_input

        
        for label, target in probing_dataset["labels"].items():
            probing_dataset["labels"][label] = np.array(target)

        self.probing_dataset = probing_dataset
                
        return probing_dataset


    def _apply_model(self, inputs, targets, label, linear = True, k_folds=10):
        """
            Apply the model over a (input, targets) pair, to test the model a cross-validation technique is used.

            Args:
                input: input dataset
                targets: target for a specific label
                linear (bool): flag to impose linearity on the probing model
                k_folds (int): number of cross validation folds

            Returns:
                loss_tr_mean (float): averaged losses over train datasets
                loss_ts_mean (float): averaged losses over test datasets
        """
        loss_tr_mean = {
            "mse": [],
            "mae": [],
            "r2" : [],
            "acc_corr": []
        }

        loss_ts_mean = {
            "mse": [],
            "mae": [],
            "r2" : [],
            "acc_corr": []
        }

        folds = KFold(n_splits=k_folds)

        splits = folds.split(np.zeros(inputs.shape[0]))

        # LOGGER.info(f"Started Cross-Validation, with K = {k_folds}")

        # do cross-validation
        for train_idx, test_idx in splits:
            train_inputs = inputs[train_idx]
            test_inputs = inputs[test_idx]
            train_targets = targets[train_idx]

            test_targets = targets[test_idx]

            # min max scaler the targets
            scaler = MinMaxScaler(feature_range=(0, 100))
            scaler.fit(train_targets.reshape(-1, 1))
            train_targets = scaler.transform(train_targets.reshape(-1, 1)).reshape(-1)
            test_targets = scaler.transform(test_targets.reshape(-1, 1)).reshape(-1)

            # train and test the model
            predicted_train = None
            predicted_test = None
            if linear:
                regr = Ridge().fit(train_inputs, train_targets)
            else:
                regr = MLPRegressor().fit(train_inputs, train_targets)

            predicted_train = regr.predict(train_inputs)
            predicted_test = regr.predict(test_inputs)

            # Mean Absolute Error
            loss_tr_mean["mae"].append(mean_absolute_error(train_targets, predicted_train))
            loss_ts_mean["mae"].append(mean_absolute_error(test_targets, predicted_test))

            # Mean Squared Error
            loss_tr_mean["mse"].append(mean_squared_error(train_targets, predicted_train))
            loss_ts_mean["mse"].append(mean_squared_error(test_targets, predicted_test))

            # R2 score
            loss_tr_mean["r2"].append(r2_score(train_targets, predicted_train))
            loss_ts_mean["r2"].append(r2_score(test_targets, predicted_test))

            if label in ["skip", "reread", "refix"]: # accuracy
                loss_tr_mean["acc_corr"].append(binary_accuracy(train_targets, predicted_train, 50))
                loss_ts_mean["acc_corr"].append(binary_accuracy(test_targets, predicted_test, 50))
            else: # correlation
                stat_tr = stats.spearmanr(train_targets, predicted_train)
                stat_ts = stats.spearmanr(test_targets, predicted_test)

                if stat_tr.pvalue < 0.05:
                    loss_tr_mean["acc_corr"].append(stat_tr.statistic)
                
                if stat_ts.pvalue < 0.05:
                loss_ts_mean["acc_corr"].append(stat_ts.statistic)

        for k in loss_tr_mean.keys():
            loss_tr_mean[k] = np.mean(loss_tr_mean[k])
            loss_ts_mean[k] = np.mean(loss_ts_mean[k])

        return inputs, targets, loss_tr_mean, loss_ts_mean


    def probe(self, linear, k_folds):
        """
            Probe the model over a built dataset.
            Saving the results in the output_dir path

            Args:
                linear (bool): flag to impose linearity on the probing model
                k_folds (int): number of cross validation folds

        """

        LOGGER.info(f"Starting probe, Linear = {linear} ...")
        metrics = dict()

        metrics["linear"] = linear



        for layer, layer_input in tqdm(self.probing_dataset["layers"].items()):
            # LOGGER.info(f"Cross Validation layer : {layer} ...")

            metrics[layer] = {}

            for label, label_target in self.probing_dataset["labels"].items():
                label_inputs, label_targets, score_train, score_test = self._apply_model(layer_input, label_target, label, linear, k_folds)

                metrics[layer].update({
                    #f"{label}_inputs" : label_inputs.tolist(),
                    #f"{label}_targets" : label_targets.tolist(),
                    f"{label}_score_train" : score_train,
                    f"{label}_score_test" : score_test
                })

                # LOGGER.info(f"Scores layer - {layer} :")
                # LOGGER.info(f"{label} train: {score_train.tolist()}")
                # LOGGER.info(f"{label} test: {score_test.tolist()}")
    
            # LOGGER.info(f"done!!!")

        with open(f"{self.output_dir}/probe_results.json", 'w') as f:
            json.dump(metrics, f)