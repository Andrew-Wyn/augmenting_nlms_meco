import torch
from anm.utils import LOGGER
from abc import ABC
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr


class Tester(ABC):
    """
        Class needed to collect train-test metrics epoch-wise.
    """
    
    def __init__(self, device, task, train_dl, test_dl):
        self.device = device
        self.task = task

        self.train_dl = train_dl
        self.test_dl = test_dl

        self.train_metrics = {}  # key-value dictionary metric --> value
        self.test_metrics = {}  # key-value dictionary metric --> value

    def evaluate(self):
        LOGGER.info("--- Evaluation Phase ---")
        LOGGER.info(f"Task: {self.task}")
        
        self.predict(self.train_dl, self.train_metrics)

        if not self.test_dl is None:
            self.predict(self.test_dl, self.test_metrics)


class GazeTester(Tester):
    """
        Implementation of Tester class for gaze features.

        Compute mse and mae metrics for each gaze feature.
    """

    def __init__(self, model, device, task, train_dl, test_dl=None):
        super().__init__(device, task, train_dl, test_dl)

        self.model = model

    def predict(self, dl, metrics):
        self.model.to(self.device)
        self.model.eval()

        metrics_ = defaultdict(lambda: 0)

        # compute the spearman correlation
        model_ouput_logits = defaultdict(list)
        gold_labels = defaultdict(list)

        with torch.no_grad():
            for batch in dl:
                # add current batch to gold_labels
                for k, v in batch["labels"].items():
                    gold_labels[k].append(v)

                batch = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": {k: v.to(self.device) for k, v in batch["labels"].items()}
                }

                model_output = self.model(**batch)

                for t, l in model_output.loss.items():
                    metrics_["mse_"+t] += l.to("cpu").numpy()

                for t, l in model_output.mae_loss.items():
                    metrics_["mae_"+t] += l.to("cpu").numpy()

                # collect model output logits
                for k, v in model_output.logits.items():
                    model_ouput_logits[k].append(v.to("cpu"))

            num_batches = len(dl)
            
            for t, l in metrics_.items():
                metrics[t] = l/num_batches

            for t in gold_labels.keys():
                model_ouput_logits_flattened = np.concatenate([a.numpy() for a in model_ouput_logits[t]]).ravel()
                gold_labels_flattened = np.concatenate([a.numpy() for a in gold_labels[t]]).ravel()

                # remove the -100 elements from the vectors
                not_masked_elements_t = gold_labels_flattened != -100

                sp = spearmanr(model_ouput_logits_flattened[not_masked_elements_t],
                                                   gold_labels_flattened[not_masked_elements_t])
                
                if sp.pvalue < 0.05:
                    metrics["spearman_"+t] = sp.statistic
                else:
                    metrics["spearman_"+t] = np.nan