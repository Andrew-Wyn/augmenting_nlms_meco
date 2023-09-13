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
    AutoConfig
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
    cf = Config.load_json("configs/modeling_configuration.json")
    config = AutoConfig.from_pretrained(model_name)
    config.update({"tasks": cf.tasks})

    model = None
    # Model
    LOGGER.info("Initiating model ...")

    if not pretrained:
        # initiate model with random weights
        # You can initialize a random BERT model using the Hugginface capabilites 
        # (from the documentation https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/bert#transformers.BertConfig)
        LOGGER.info("Take randomized model")

        if model_type == "Roberta":
            model = RobertaForMultiTaskTokenClassification(config)
        elif model_type == "XLM":
            model = XLMRobertaForMultiTaskTokenClassification(config)
        elif model_type == "Camembert":
            model = CamembertForMultiTaskTokenClassification(config)
    else:
        LOGGER.info("Take pretrained model")

        if model_type == "Roberta":
            model = RobertaForMultiTaskTokenClassification.from_pretrained(model_name, config=config)
        elif model_type == "XLM":
            model = XLMRobertaForMultiTaskTokenClassification.from_pretrained(model_name, config=config)
        elif model_type == "Camembert":
            model = CamembertForMultiTaskTokenClassification.from_pretrained(model_name, config=config)

    return model