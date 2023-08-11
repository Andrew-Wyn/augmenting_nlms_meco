import torch
import torch.nn as nn

from anm.gaze_training.utils import mask_mse_loss, GazePredictionLoss
from anm.utils import LOGGER
from abc import ABC, abstractmethod

class Tester(ABC):
    def __init__(self, quantities, device, task, train_dl, test_dl):
        self.quantities = quantities  # list of metrics to be evaluated (other than the loss)
        self.device = device
        self.task = task

        self.train_dl = train_dl
        self.test_dl = test_dl

        self.preds = []
        self.logs = []
        self.maes = []
        self.train_metrics = {}  # key-value dictionary metric --> value
        self.test_metrics = {}  # key-value dictionary metric --> value
        self.train_units = {}  # key-value dictionary metric --> measurement unit
        self.test_units = {}  # key-value dictionary metric --> measurement unit

    def evaluate(self):
        # LOGGER.info(f"Begin evaluation task {self.task}")
        self.predict(self.train_dl, self.train_metrics, self.train_units)

        if not self.test_dl is None:
            self.predict(self.test_dl, self.test_metrics, self.test_units)

        # for key in self.train_metrics:
        #     LOGGER.info(f"train_{key}: {self.train_metrics[key]:.4f} {self.train_units[key]}")

        # for key in self.test_metrics:
        #     LOGGER.info(f"test_{key}: {self.test_metrics[key]:.4f} {self.test_units[key]}")

    @abstractmethod
    def predict(self):
        pass


class GazeTester(Tester):
    def __init__(self, model, device, task, train_dl, test_dl=None):
        quantities = []
        super().__init__(quantities, device, task, train_dl, test_dl)

        self.model = model
        self.target_pad = train_dl.target_pad

        self.criterion = nn.MSELoss(reduction="mean")
        self.criterion_metric = GazePredictionLoss(model.num_labels)

    def predict(self, dl, metrics, units):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            loss = 0
            losses_metric = torch.zeros(self.criterion_metric.d_report)
            self.preds = []

            for batch in dl:
                b_input, b_target, b_mask = batch
                b_input = b_input.to(self.device)
                b_target = b_target.to(self.device)
                b_mask = b_mask.to(self.device)

                b_output = self.model(input_ids=b_input, attention_mask=b_mask)[0]

                active_outputs, active_targets = mask_mse_loss(b_output, b_target, self.target_pad, self.model.num_labels)
                loss += self.criterion(active_outputs, active_targets)

                b_output_orig_len = []
                b_target_orig_len = []
                for output, target in zip(b_output, b_target):
                    active_idxs = (target != self.target_pad)[:, 0]
                    b_output_orig_len.append(output[active_idxs])
                    b_target_orig_len.append(target[active_idxs])

                losses_metric += self.criterion_metric(b_output_orig_len, b_target_orig_len)

                self.preds.extend([i.cpu().numpy() for i in b_output_orig_len])

            num_batches = len(dl)
            loss /= num_batches
            losses_metric /= num_batches

            metrics["loss"] = loss.item()
            units["loss"] = ""
            metrics["loss_all"] = losses_metric[0].item()
            units["loss_all"] = ""
            for i, value in enumerate(losses_metric[1:]):
                metrics["loss_" + str(i)] = value.item()
                units["loss_" + str(i)] = ""

    def calc_metrics(self):
        pass