import os
import torch
import torch.nn as nn
from tqdm import tqdm
from abc import ABC, abstractmethod
from anm.gaze_training.tester import GazeTester
from anm.utils import LOGGER
from anm.gaze_training.utils import mask_mse_loss


class Trainer(ABC):
    def __init__(self, cf, model, train_dl, tester, task, device, writer):
        self.model = model
        self.train_dl = train_dl
        self.n_epochs = cf.n_epochs
        self.task = task
        self.device = device
        self.writer = writer
        self.tester = tester
        self.cf = cf

    @abstractmethod
    def train_one_step(self, batch):
        pass

    def train(self, save_model=False, output_dir=None):
        n_batches_one_epoch = len(self.train_dl)
        n_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f"Num epochs: {self.n_epochs}")
        LOGGER.info(f"Num parameters: {n_params}")
        LOGGER.info(f"Begin training task {self.task}")

        self.model.to(self.device)
        self.model.train()

        it = 1

        for _ in tqdm(range(1, self.n_epochs + 1)):
            for batch in self.train_dl:
                it += 1

                loss = self.train_one_step(batch)
                self.writer.add_scalar(f"{self.task}/train/loss_step_wise", loss, it)

            self.tester.evaluate()

            for key, metric in self.tester.train_metrics.items():
                self.writer.add_scalar(f"{self.task}/train/{key}", metric, it // n_batches_one_epoch)
            
            if not self.tester.test_dl is None: 
                for key, metric in self.tester.test_metrics.items():
                    self.writer.add_scalar(f"{self.task}/test/{key}", metric, it // n_batches_one_epoch)

        LOGGER.info(f"Training Done -> Train Loss_all : {self.tester.train_metrics['loss_all']}")
        if not self.tester.test_dl is None:
            LOGGER.info(f"Training Done -> Test Loss_all : {self.tester.test_metrics['loss_all']}")

        # save the model after last epoch
        if save_model:
            folder_name = os.path.join(output_dir, "model-"+self.cf.model_name+"-finetuned")
            
            if self.cf.random_weights:
                folder_name = folder_name + "-randomized"
            else:
                folder_name = folder_name + "-pretrained"

            if self.cf.full_finetuning:
                folder_name = folder_name + "-full"
            else:
                folder_name = folder_name + "-notfull"

            self.model.save_pretrained(folder_name)


class GazeTrainer(Trainer):
    def __init__(self, cf, model, train_dl, optim, scheduler,
                 task, device, writer, test_dl=None):
        tester = GazeTester(model, device, task, train_dl, test_dl)
        super().__init__(cf, model, train_dl, tester, task, device, writer)

        self.optim = optim
        self.scheduler = scheduler
        self.max_grad_norm = cf.max_grad_norm
        self.target_pad = train_dl.target_pad

        self.criterion = nn.MSELoss(reduction="mean")

    def train_one_step(self, batch):
        self.model.zero_grad()

        b_input, b_target, b_mask = batch
        b_input = b_input.to(self.device)
        b_target = b_target.to(self.device)
        b_mask = b_mask.to(self.device)

        # forward pass over one batch
        b_output = self.model(input_ids=b_input, attention_mask=b_mask)[0]
        
        # compute loss
        active_outputs, active_targets = mask_mse_loss(b_output, b_target, self.target_pad, self.model.num_labels)
        loss = self.criterion(active_outputs, active_targets)

        # backward pass over one batch
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)

        self.optim.step()
        self.scheduler.step()

        return loss.item()