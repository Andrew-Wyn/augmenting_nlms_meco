# TODO: follow the camembert.py file

# CODE MODIFIED FROM https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model."""

from typing import Optional, Tuple, Union
from anm.modeling.utils import mask_loss

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from dataclasses import dataclass

from transformers.utils import logging, ModelOutput

from transformers import RobertaPreTrainedModel, RobertaModel

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

@dataclass
class MultiTaskTokenClassifierOutput(ModelOutput):
    """
    Class for outputs of multitask token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    binary_loss: Optional[dict] = None
    continuous_loss: Optional[dict] = None
    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForMultiTaskTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # classifiers
        self.tasks = ['skip', 'firstfix-dur', 'firstrun-dur', 'dur', 'firstrun-nfix', 'nfix', 'refix', 'reread']
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, 1) for
            task in self.tasks
        })

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None, # cambiare tipo
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultiTaskTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        binary_loss = {}
        continuous_loss = {}
        logits = None

        for task in self.tasks:
            task_logits = self.classifiers[task](sequence_output)
            if labels[task] is not None:
                task_labels = labels[task].to(task_logits.device)
                # TODO: mask out the output associated with not-first-token of a word
                # ERROR: check dimensionalities
                output_, target_ = mask_loss(task_logits, task_labels, -100)

                if task in ['skip', 'refix', 'reread']:
                    loss_fct = BCEWithLogitsLoss()
                    binary_loss[task] = loss_fct(output_, target_)
                else:
                    loss_fct = MSELoss()
                    continuous_loss[task] = torch.sqrt(loss_fct(output_, target_))

        # if not return_dict:                               Se serve bisogna sistemare i logits perchè hanno dimensioni diverse
        #     logits = torch.stack(logits, dim=-1)
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return MultiTaskTokenClassifierOutput(
            binary_loss=binary_loss,
            continuous_loss=continuous_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
