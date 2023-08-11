from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import logging.config
import torch.nn.functional as F
import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# TODO: wait to merge issue#4
# from modeling.multioutput_xlm_roberta import XLMRobertaMultiTaskForSequenceRegression
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
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


def create_finetuning_optimizer(cf, model):
    """
    Creates an Adam optimizer with weight decay. We can choose whether to perform full finetuning on
    all parameters of the model or to just optimize the parameters of the final classification layer.
    """
    if cf.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay_rate": cf.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay_rate": 0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for _, p in param_optimizer]}]

    return AdamW(optimizer_grouped_parameters, lr=cf.lr, eps=cf.eps)


def create_scheduler(cf, optim, dl):
    """
    Creates a linear learning rate scheduler.
    """
    n_iters = cf.n_epochs * len(dl)
    return get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=n_iters)


def minMaxScaling(train_targets, test_targets=None, feature_max=1, pad_token=-1):
    # generate train set for the scaler.
    scaler = MinMaxScaler(feature_range=[0, feature_max])

    flat_features = []
    for sentence_taget in train_targets:
        for token_target in sentence_taget:
            if np.all(token_target != pad_token):
                flat_features.append(token_target)

    scaler.fit(flat_features)

    train_targets_ret = []

    for sentence_taget in train_targets:
        train_sentence_target_ret = []
        for token_target in sentence_taget:
            if np.all(token_target != pad_token):
                train_sentence_target_ret.append(scaler.transform([token_target])[0])
            else:
                train_sentence_target_ret.append(token_target)
        train_targets_ret.append(train_sentence_target_ret)

    if not test_targets is None:
        test_targets_ret = []
        for sentence_taget in test_targets:
            test_sentence_target_ret = []
            for token_target in sentence_taget:
                if np.all(token_target != pad_token):
                    test_sentence_target_ret.append(scaler.transform([token_target])[0])
                else:
                    test_sentence_target_ret.append(token_target)
            test_targets_ret.append(test_sentence_target_ret)

        return train_targets_ret, test_targets_ret

    return train_targets_ret


def load_model_from_hf(model_name, pretrained, multiregressor, d_out=8):

    # Model
    LOGGER.info("Initiating model ...")
    if not pretrained:
        # initiate model with random weights
        LOGGER.info("Take randomized model")
        model = None
    else:
        LOGGER.info("Take pretrained model")
        model = None
    return model


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