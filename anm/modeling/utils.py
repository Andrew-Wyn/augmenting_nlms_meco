import torch
from torch.nn import MSELoss, L1Loss
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import logging, ModelOutput

def mask_loss(b_output, b_target, target_pad):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_outputs = b_output.view(-1)
    active_targets = b_target.view(-1)
    active_mask = active_targets == target_pad

    active_outputs = torch.where(active_mask, active_targets, active_outputs)

    return active_outputs, active_targets


def gaze_multitask_forward(tasks, classifiers, sequence_output, labels):
    loss = {}
    mae_loss = {}
    logits = list()

    for task in tasks:
        task_logits = classifiers[task](sequence_output)
        logits.append(task_logits)
        if labels[task] is not None:
            task_labels = labels[task].to(task_logits.device)
            # TODO: mask out the output associated with not-first-token of a word
            # ERROR: check dimensionalities
            output_, target_ = mask_loss(task_logits, task_labels, -100)
            
            # MSE Loss
            loss_fct = MSELoss()
            loss[task] = loss_fct(output_, target_)

            with torch.no_grad():
                # MAE Loss
                loss_fct = L1Loss()
                mae_loss[task] = loss_fct(output_, target_)

    return loss, mae_loss, logits


@dataclass
class MultiTaskTokenClassifierOutput(ModelOutput):
    """
    Class for outputs of multitask token classification models.

    Args:
        loss (dict) :
            MSE loss.
        mae_loss (dict) :
            MAE loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the selfattention
            heads.
    """

    loss: Optional[dict] = None
    mae_loss: Optional[dict] = None
    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None