from transformers import get_linear_schedule_with_warmup
import torch
import logging.config
import torch.nn.functional as F
import json
from sklearn.preprocessing import MinMaxScaler
from anm.modeling.multitask_roberta import RobertaForMultiTaskTokenClassification
from anm.modeling.multitask_xlm_roberta import XLMRobertaForMultiTaskTokenClassification
from anm.modeling.multitask_camembert import CamembertForMultiTaskTokenClassification
import numpy as np
from transformers import (
    RobertaConfig,
    XLMRobertaConfig,
    CamembertConfig
)

CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "processing": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}


logging.config.dictConfig(CONFIG)
LOGGER = logging.getLogger("processing")


class Config:
    @classmethod
    def load_json(cls, fpath):
        cf = cls()
        with open(fpath) as f:
            cf.__dict__ = json.load(f)

        return cf


def create_scheduler(cf, optim, dl):
    """
    Creates a linear learning rate scheduler.
    """
    n_iters = cf.n_epochs * len(dl)
    return get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=n_iters)


def load_model_from_hf(model_type, model_name, pretrained):
    model = None
    # Model
    LOGGER.info("Initiating model ...")

    if not pretrained:
        # initiate model with random weights
        # You can initialize a random BERT model using the Hugginface capabilites 
        # (from the documentation https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/bert#transformers.BertConfig)
        LOGGER.info("Take randomized model")

        if model_type == "Roberta":
            config = RobertaConfig()
            model = RobertaForMultiTaskTokenClassification(config)
        elif model_type == "XLM":
            config = XLMRobertaConfig()
            model = XLMRobertaForMultiTaskTokenClassification(config)
        elif model_type == "Camembert":
            config = CamembertConfig()
            model = CamembertForMultiTaskTokenClassification(config)
    else:
        LOGGER.info("Take pretrained model")

        if model_type == "Roberta":
            model = RobertaForMultiTaskTokenClassification.from_pretrained(model_name)
        elif model_type == "XLM":
            model = XLMRobertaForMultiTaskTokenClassification.from_pretrained(model_name)
        elif model_type == "Camembert":
            model = CamembertForMultiTaskTokenClassification.from_pretrained(model_name)

    return model


# TODO: move under attn_correlation module, used by ALTI
def normalize_contributions(model_contributions,scaling='minmax',resultant_norm=None):
    """Normalization of the matrix of contributions/weights extracted from the model."""

    normalized_model_contributions = torch.zeros(model_contributions.size())
    for l in range(0,model_contributions.size(0)):

        if scaling == 'min_max':
            ## Min-max normalization
            min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
            max_importance_matrix = model_contributions[l].max(-1, keepdim=True)[0]
            normalized_model_contributions[l] = (model_contributions[l] - min_importance_matrix) / (max_importance_matrix - min_importance_matrix)
            normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)

        elif scaling == 'sum_one':
            normalized_model_contributions[l] = model_contributions[l] / model_contributions[l].sum(dim=-1,keepdim=True)
            #normalized_model_contributions[l] = normalized_model_contributions[l].clamp(min=0)

        # For l1 distance between resultant and transformer vectors we apply min_sum
        elif scaling == 'min_sum':
            if resultant_norm == None:
                min_importance_matrix = model_contributions[l].min(-1, keepdim=True)[0]
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(min_importance_matrix)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
            else:
                # print('resultant_norm[l]', resultant_norm[l].size())
                # print('model_contributions[l]', model_contributions[l])
                # print('normalized_model_contributions[l].sum(dim=-1,keepdim=True)', model_contributions[l].sum(dim=-1,keepdim=True))
                normalized_model_contributions[l] = model_contributions[l] + torch.abs(resultant_norm[l].unsqueeze(1))
                normalized_model_contributions[l] = torch.clip(normalized_model_contributions[l],min=0)
                normalized_model_contributions[l] = normalized_model_contributions[l] / normalized_model_contributions[l].sum(dim=-1,keepdim=True)
        elif scaling == 'softmax':
            normalized_model_contributions[l] = F.softmax(model_contributions[l], dim=-1)
        elif scaling == 't':
            model_contributions[l] = 1/(1 + model_contributions[l])
            normalized_model_contributions[l] =  model_contributions[l]/ model_contributions[l].sum(dim=-1,keepdim=True)
        else:
            print('No normalization selected!')
    return normalized_model_contributions