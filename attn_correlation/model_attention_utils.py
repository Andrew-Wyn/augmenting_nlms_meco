from utils.explainability.ValueZeroing.modeling.customized_modeling_bert import BertForMaskedLM
from utils.explainability.ValueZeroing.modeling.customized_modeling_roberta import RobertaForMaskedLM
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoConfig
import numpy as np
import torch

device = torch.device("cpu")


class ValueZeroingContributions:

    def __init__(self, model_name, layer=-1, rollout=True):
        self.model = self.load_model(model_name)
        self.layer = layer
        self.rollout = rollout

    def load_model(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        if 'roberta' in model_name:
            model = RobertaForMaskedLM.from_pretrained(model_name, config=config)
        else:
            model = BertForMaskedLM.from_pretrained(model_name, config=config)
        return model

    def __compute_joint_attention(self, att_mat, res=True):
        if res:
            residual_att = np.eye(att_mat.shape[1])[None, ...]
            att_mat = att_mat + residual_att
            att_mat = att_mat / att_mat.sum(axis=-1)[..., None]

        joint_attentions = np.zeros(att_mat.shape)
        layers = joint_attentions.shape[0]
        joint_attentions[0] = att_mat[0]
        for i in np.arange(1, layers):
            joint_attentions[i] = att_mat[i].dot(joint_attentions[i - 1])

        return joint_attentions

    def get_contributions(self, tokenized_text):
        tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
        with torch.no_grad():
            try:
                outputs = self.model(tokenized_text['input_ids'],
                                     attention_mask=tokenized_text['attention_mask'],
                                     token_type_ids=tokenized_text['token_type_ids'],
                                     output_hidden_states=True, output_attentions=False)
            except:
                outputs = self.model(tokenized_text['input_ids'],
                                     attention_mask=tokenized_text['attention_mask'],
                                     output_hidden_states=True, output_attentions=False)
        org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)

        input_shape = tokenized_text['input_ids'].size()
        batch_size, seq_length = input_shape

        score_matrix = np.zeros((self.model.config.num_hidden_layers, seq_length, seq_length))
        try:
            layers_modules = self.model.bert.encoder.layer
        except:
            layers_modules = self.model.roberta.encoder.layer
        for l, layer_module in enumerate(layers_modules):
            # print(l)
            # print('\n____________________________________________\n')
            for t in range(seq_length):
                try:
                    extended_blanking_attention_mask: torch.Tensor = self.model.bert.get_extended_attention_mask(
                        tokenized_text['attention_mask'], input_shape, device=device)
                except:
                    extended_blanking_attention_mask: torch.Tensor = self.model.roberta.get_extended_attention_mask(
                        tokenized_text['attention_mask'], input_shape, device=device)

                with torch.no_grad():
                    layer_outputs = layer_module(org_hidden_states[l].unsqueeze(0),  # previous layer's original output
                                                 attention_mask=extended_blanking_attention_mask,
                                                 output_attentions=False,
                                                 zero_value_index=t,
                                                 )

                hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
                # compute similarity between original and new outputs
                # cosine
                x = hidden_states
                y = org_hidden_states[l + 1].detach().cpu().numpy()
                distances = cosine_distances(x, y).diagonal()

                score_matrix[l, :, t] = distances
        valuezeroing_scores = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True)


        if not self.rollout:
            return valuezeroing_scores
        rollout_valuezeroing_scores = self.__compute_joint_attention(valuezeroing_scores, res=False)
        return rollout_valuezeroing_scores


def normalize_contribs(contribs):
    return (contribs - np.min(contribs)) / (np.max(contribs) - np.min(contribs))


def compute_aggregation(contribs, method):
    agg_contribs = []
    agg_el = None
    for el in contribs:
        if method == 'mean':
            agg_el = sum(el) / len(el)
        elif method == 'first':
            agg_el = el[0]
        elif method == 'sum':
            agg_el = sum(el)
        elif method == 'max':
            agg_el = max(el)
        agg_contribs.append(agg_el)
    return agg_contribs


def aggregate_subtokens_contribs(contribs, al_ids, agg_method, normalize=False):
    contribs = contribs[1: -1]

    agg_contribs = [[] for _ in range(len(set(al_ids)))]
    # if normalize:
    #     contribs = normalize_contribs(contribs)
    for al_id, contrib in zip(al_ids, contribs):
        agg_contribs[al_id].append(contrib)
    agg_contribs = compute_aggregation(agg_contribs, agg_method)
    return agg_contribs
