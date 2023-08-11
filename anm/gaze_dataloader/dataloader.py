import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class GazeDataLoader(DataLoader):
    def __init__(self, cf, inputs, targets, masks, target_pad, mode):

        self.target_pad = target_pad

        dataset = TensorDataset(torch.as_tensor(inputs),
                                torch.as_tensor(targets, dtype=torch.float32),
                                torch.as_tensor(masks))
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        batch_size = cf.train_bs if mode == "train" else cf.eval_bs

        super().__init__(dataset, sampler=sampler, batch_size=batch_size)