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

from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from anm.modeling.utils import gaze_multitask_forward, MultiTaskTokenClassifierOutput, MultiTaskSequenceClassifierOutput

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.utils import logging

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


class RobertaForMultiTaskTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # classifiers
        self.tasks = config.tasks
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
            labels: Optional[torch.LongTensor] = None,  # cambiare tipo
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

        loss, mae_loss, logits = gaze_multitask_forward(self.tasks, self.classifiers, sequence_output, labels)

        # if not return_dict:                               Se serve bisogna sistemare i logits perchè hanno dimensioni diverse
        #     logits = torch.stack(logits, dim=-1)
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return MultiTaskTokenClassifierOutput(
            loss=loss,
            mae_loss=mae_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultiTaskSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.tasks = config.tasks
        self.classifiers = nn.ModuleDict({
            task: RobertaClassificationHead(config) for
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
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultiTaskSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

        loss = None
        logits = dict()
        loss_fct = CrossEntropyLoss()
        for task in self.tasks:
            logits[task] = self.classifiers[task](sequence_output)
            if labels[task] is not None:
                task_loss = loss_fct(logits[task].view(-1, 2), labels[task].view(-1))
                loss = task_loss if loss is None else loss+task_loss
        loss /= len(self.tasks)

        
        return MultiTaskSequenceClassifierOutput(
            loss=loss,
            tasks_loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
