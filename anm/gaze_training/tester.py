import torch
from anm.utils import LOGGER
from abc import ABC
from collections import defaultdict

class Tester(ABC):
    def __init__(self, device, task, train_dl, test_dl):
        self.device = device
        self.task = task

        self.train_dl = train_dl
        self.test_dl = test_dl

        self.train_metrics = defaultdict(lambda: 0)  # key-value dictionary metric --> value
        self.test_metrics = defaultdict(lambda: 0)  # key-value dictionary metric --> value

    def evaluate(self):
        # LOGGER.info(f"Begin evaluation task {self.task}")
        self.predict(self.train_dl, self.train_metrics)

        if not self.test_dl is None:
            self.predict(self.test_dl, self.test_metrics)


class GazeTester(Tester):
    def __init__(self, model, device, task, train_dl, test_dl=None):
        super().__init__(device, task, train_dl, test_dl)

        self.model = model

    def predict(self, dl, metrics):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in dl:
                model_output = self.model(**batch)

                for t, l in model_output.loss.items():
                    metrics["mse_"+t] += l

                for t, l in model_output.mae_loss.items():
                    metrics["mae_"+t] += l

            num_batches = len(dl)
            
            for t, l in metrics.items():
                metrics[t] = l/num_batches