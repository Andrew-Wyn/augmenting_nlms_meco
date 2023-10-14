import numpy as np
from anm.utils import LOGGER
from collections import defaultdict
from tqdm import tqdm
import torch
import datetime
import json

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from tqdm import tqdm


class DownstreamProber():
    """
        Class used to compute the probing experiments
    """

    def __init__(self, train_dataloader, test_dataloader, output_dir):
        """
            Args:
                dataloader: the dataloader of the dataset
                output_dir (str): the path where the resutls will be saved
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.probing_dataset_train = None
        self.probing_dataset_test = None
        self.output_dir = output_dir

    def create_probing_dataset(self, model, tokenizer):
        """
            Create probing dataset, given the model to probe.

            Args:
                model: the model over which compute the probing
            Returns:
                probing_dataset (dict): the probing dataset, with a structure defined by us.
                {"layers": {"i": np.array}, "labels": {"label_name": []}}
        """
        LOGGER.info(f"Creating probing datasets...")

        probing_dataset_train = {
            "layers": {}, # for each layer it contains the hidden representations of each token (first sub-token)
            "label": [] # it contains the list of the target values
        }

        probing_dataset_test = {
            "layers": {}, # for each layer it contains the hidden representations of each token (first sub-token)
            "label": [] # it contains the list of the target values
        }

        # cicle over the dataloader, one batch at time, assumed to have batch_size = 1
        for batch in tqdm(list(self.train_dataloader)[:]):
            # print(batch)
            probing_dataset_train["label"].append(batch["labels"].numpy())

            # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(batch["input_ids"][0])))
                                    
            # pass the input in the model to retrieve the hidden states
            with torch.no_grad():
                model_output = model(**batch, output_hidden_states=True)

            # for each layer fill the relative storage in the returning dict, taking only the first sub-token elements
            for layer in range(model.config.num_hidden_layers):

                hidden_state = model_output.hidden_states[layer].numpy()
                    
                probe_input = hidden_state[0, 0, :] #first dimension: batch = 1

                if layer in probing_dataset_train["layers"].keys():
                    probing_dataset_train["layers"][layer].append(probe_input)
                else:
                    probing_dataset_train["layers"][layer] = [probe_input]


        self.probing_dataset_train = probing_dataset_train

        for batch in tqdm(list(self.test_dataloader)[:]):
            # print(batch)
            probing_dataset_test["label"].append(batch["labels"].numpy())

            # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(batch["input_ids"][0])))
                                    
            # pass the input in the model to retrieve the hidden states
            with torch.no_grad():
                model_output = model(**batch, output_hidden_states=True)

            # for each layer fill the relative storage in the returning dict, taking only the first sub-token elements
            for layer in range(model.config.num_hidden_layers):

                hidden_state = model_output.hidden_states[layer].numpy()
                    
                probe_input = hidden_state[0, 0, :] #first dimension: batch = 1

                if layer in probing_dataset_test["layers"].keys():
                    probing_dataset_test["layers"][layer].append(probe_input)
                else:
                    probing_dataset_test["layers"][layer] = [probe_input]

        self.probing_dataset_test = probing_dataset_test

        return probing_dataset_test, probing_dataset_test


    def _apply_model(self, train_inputs, train_targets, test_inputs, test_targets, linear = True):
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

        # train and test the model
        predicted_train = None
        predicted_test = None

        if linear:
            regr = RidgeClassifier().fit(train_inputs, train_targets)
        else:
            regr = MLPClassifier().fit(train_inputs, train_targets)

        predicted_train = regr.predict(train_inputs)
        predicted_test = regr.predict(test_inputs)

        loss_tr = accuracy_score(train_targets, predicted_train)

        loss_ts = accuracy_score(test_targets, predicted_test)

        return loss_tr, loss_ts


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

        for (layer, layer_input_tr), layer_input_ts in tqdm(zip(self.probing_dataset_train["layers"].items(), \
                                                        self.probing_dataset_test["layers"].values())):
            # LOGGER.info(f"Cross Validation layer : {layer} ...")

            tr_target = self.probing_dataset_train["label"]
            ts_target = self.probing_dataset_test["label"]

            metrics[layer] = {}

            loss_tr, loss_ts = self._apply_model(layer_input_tr, tr_target, layer_input_ts, ts_target)

            metrics[layer]["loss_tr"] = loss_tr
            metrics[layer]["loss_ts"] = loss_ts

            # LOGGER.info(f"Scores layer - {layer} :")
            # LOGGER.info(f"train: {score_train.tolist()}")
            # LOGGER.info(f"test: {score_test.tolist()}")
    
            # LOGGER.info(f"done!!!")

        with open(f"{self.output_dir}/probe_results.json", 'w') as f:
            json.dump(metrics, f)