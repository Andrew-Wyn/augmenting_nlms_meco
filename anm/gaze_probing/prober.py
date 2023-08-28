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
from sklearn.metrics import mean_absolute_error


class Prober():
    def __init__(self, dataloader, feature_max, output_dir):
        self.dataloader = dataloader
        self.probing_dataset = None
        self.feature_max = feature_max
        self.output_dir = output_dir

    def create_probing_dataset(self, model):
        LOGGER.info(f"Creating datasets...")

        probing_dataset = defaultdict(list)

        print(model.config.num_hidden_layers)

        LOGGER.info(f"Start creating dataset...")

        for batch in self.dataloader:

            with torch.no_grad():
                model_output = model(**batch)

                print(model_output)

                exit()
            
            for layer in range(model.config.num_hidden_layers):

                hidden_state = model_output.hidden_states[layer].numpy()

                non_masked_els = np.multiply.reduce(target != -1, 1) > 0

                probe_input = hidden_state[0, non_masked_els, :]

                output = target[non_masked_els, :]

                # take elements token-wise
                for i in range(probe_input.shape[0]):
                    probing_dataset[layer].append((probe_input[i], output[i]))
                
        LOGGER.info("Retrieving done, postprocess...")
        
        # concatenate the inputs and outputs !!!!
        for layer in range(model.config.num_hidden_layers):
            input_list = []
            output_list = []
            
            for probe_input, output in probing_dataset[layer]:
                input_list.append(probe_input)
                output_list.append(output)

            probing_dataset[layer] = (input_list, output_list)

        self.probing_dataset = probing_dataset
        
        return probing_dataset


    def _apply_model(self, inputs, targets, linear = True, k_folds=10):
        # do cross-validation

        l = len(inputs)
        l_ts = l//k_folds

        loss_tr_mean = None
        loss_ts_mean = None

        for k in tqdm(range(k_folds)):
            # cicle over folds, for every fold create train_d, test_d
            if k != k_folds-1: # exclude the k-th part from the validation
                train_inputs = inputs[:(k)*l_ts] + inputs[(k+1)*l_ts:]
                train_targets = targets[:(k)*l_ts] + targets[(k+1)*l_ts:]
                test_inputs = inputs[k*l_ts:(k+1)*l_ts]
                test_targets = targets[k*l_ts:(k+1)*l_ts]

            else: # last fold clausole
                train_inputs = inputs[:k*l_ts]
                train_targets = targets[:k*l_ts]
                test_inputs = inputs[k*l_ts:]
                test_targets = targets[k*l_ts:]

            # min max scaler the targets
            scaler = MinMaxScaler(feature_range=[0, self.feature_max])
            scaler.fit(train_targets)
            train_targets = scaler.transform(train_targets)
            test_targets = scaler.transform(test_targets)

            # apply a model for each feature
            predicted_train = None
            predicted_test = None
            # learn a model for each feature, then concatenate the predictions, 
            for feat_i in range(train_targets.shape[1]):
                if linear:
                    # regr = SVR(kernel="linear", degree=1).fit(train_inputs, train_targets[:, feat_i])
                    regr = Ridge().fit(train_inputs, train_targets[:, feat_i])
                else:
                    regr = MLPRegressor().fit(train_inputs, train_targets[:, feat_i])

                if predicted_train is None:
                    predicted_train = np.expand_dims(regr.predict(train_inputs), axis=0)
                    predicted_test = np.expand_dims(regr.predict(test_inputs), axis=0)
                else:
                    predicted_train = np.concatenate((predicted_train, np.expand_dims(regr.predict(train_inputs), axis=0)), axis=0)
                    predicted_test = np.concatenate((predicted_test, np.expand_dims(regr.predict(test_inputs), axis=0)), axis=0)

            predicted_train = predicted_train.T
            predicted_test = predicted_test.T

            # Train errors
            loss_tr = np.concatenate((([mean_absolute_error(train_targets, predicted_train)], mean_absolute_error(train_targets, predicted_train, multioutput='raw_values'))), axis=0)

            if not loss_tr_mean is None:
                loss_tr_mean += loss_tr
            else:
                loss_tr_mean = loss_tr

            # Test errors
            loss_ts = np.concatenate(([mean_absolute_error(test_targets, predicted_test)], mean_absolute_error(test_targets, predicted_test, multioutput='raw_values')), axis=0)

            if not loss_ts_mean is None:
                loss_ts_mean += loss_ts
            else:
                loss_ts_mean = loss_ts

        loss_tr_mean /= k_folds
        loss_ts_mean /= k_folds

        return loss_tr_mean, loss_ts_mean


    def probe(self, linear, k_folds):
        LOGGER.info(f"Starting probe, Linear = {linear} ...")
        metrics = dict()

        metrics["linear"] = linear

        for layer, dataset in self.probing_dataset.items():
            LOGGER.info(f"Cross Validation layer : {layer} ...")

            inputs, targets = dataset

            score_train, score_test = self._apply_model(inputs, targets, linear, k_folds)

            metrics[layer] = {
                "score_train" : score_train.tolist(),
                "score_test" : score_test.tolist()
            }

            LOGGER.info(f"Scores layer - {layer} :")
            LOGGER.info(f"Train: {score_train.tolist()}")
            LOGGER.info(f"Test: {score_test.tolist()}")
            LOGGER.info(f"done!!!")

        with open(f"{self.output_dir}/probe_results.json", 'w') as f:
            json.dump(metrics, f)