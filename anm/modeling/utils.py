import torch

def mask_loss(b_output, b_target, target_pad):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_outputs = b_output.view(-1)
    active_targets = b_target.view(-1)
    active_mask = active_targets == target_pad

    active_outputs = torch.where(active_mask, active_targets, active_outputs)

    return active_outputs, active_targets