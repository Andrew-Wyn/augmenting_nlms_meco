# CODE MODIFIED FROM https://github.com/huggingface/transformers/blob/main/src/transformers/models/camembert/modeling_camembert.py

# coding=utf-8
# Copyright 2019 Inria, Facebook AI Research and the HuggingFace Inc. team.
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
"""PyTorch CamemBERT model."""

from typing import Optional, Tuple, Union

import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass

from transformers.utils import logging, ModelOutput

from transformers import CamembertPreTrainedModel, CamembertModel


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "camembert-base"
_CONFIG_FOR_DOC = "CamembertConfig"

CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "camembert-base",
    "Musixmatch/umberto-commoncrawl-cased-v1",
    "Musixmatch/umberto-wikipedia-uncased-v1",
    # See all CamemBERT models at https://huggingface.co/models?filter=camembert
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

    loss: Optional[torch.FloatTensor] = None
    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.roberta.modeling_roberta.RobertaForTokenClassification with Roberta->Camembert, ROBERTA->CAMEMBERT
class CamembertForMultiTaskTokenClassification(CamembertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = CamembertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # classifiers
        tasks = ['skip', 'firstfix_dur', 'firstrun_dur', 'dur', 'firstrun_nfix', 'nfix', 'refix', 'reread']
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, 2 if task in ['skip', 'refix', 'reread'] else 1) for
            task in tasks
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
        labels: Optional[torch.LongTensor] = None,
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

        tasks = ['skip', 'firstfix_dur', 'firstrun_dur', 'dur', 'firstrun_nfix', 'nfix', 'refix', 'reread']
        loss = None
        logits = None

        for task in tasks:
            task_logits = self.classifiers[task](sequence_output)
            if labels[task] is not None:
                task_labels = labels[task].to(task_logits.device)
                if task in ['skip', 'refix', 'reread']:
                    loss_fct = CrossEntropyLoss()
                    task_loss = loss_fct(task_logits.view(-1, 2), task_labels.view(-1))
                else:
                    loss_fct = MSELoss()
                    task_loss = loss_fct(task_logits.squeeze(), task_labels.squeeze())
                if loss is None:
                    loss = task_loss
                else:
                    loss += task_loss

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return MultiTaskTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )