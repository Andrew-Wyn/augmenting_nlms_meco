import torch
from transformers import AdamW


def mask_mse_loss(b_output, b_target, target_pad, d_out):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_mask = b_target.view(-1, d_out) == target_pad
    active_outputs = b_output.view(-1, d_out)
    active_targets = torch.where(active_mask, active_outputs, b_target.view(-1, d_out))

    return active_outputs, active_targets
    

def create_finetuning_optimizer(cf, model):
    """
    Creates an Adam optimizer with weight decay.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay_rate": cf.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay_rate": 0}
    ]

    return AdamW(optimizer_grouped_parameters, lr=cf.lr, eps=cf.eps)
