from anm.attn_correlation.modules.ValueZeroing.modeling.customized_modeling_xlm_roberta import XLMRobertaForMaskedLMVZ
from anm.attn_correlation.modules.ValueZeroing.modeling.customized_modeling_camembert import CamembertForMaskedLMVZ
from anm.attn_correlation.modules.ValueZeroing.modeling.customized_modeling_roberta import RobertaForMaskedLMVZ
from anm.attn_correlation.modules.alti.src.contributions import ModelWrapper
from anm.attn_correlation.modules.alti.src.utils_contributions import *
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os


class TokenContributionExtractor(ABC):

    def __init__(self, model_name: str, layer: int, rollout: bool, aggregation_method: str, random_init: bool = False):
        self.random_init = random_init
        self.layer = layer
        self.rollout = rollout
        self.aggregration_method = aggregation_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_name)

    @abstractmethod
    def _load_model(self, model_name: str):
        """
        Left abstract since some contribution extraction methods have to modify the default model architecture
        """

    @abstractmethod
    def compute_sentence_contributions(self, tokenized_text):
        """
        Compute contributions given a tokenized sentence
        """

    def _compute_aggregation(self, contributions):
        aggregated_contributions = []
        aggregated_element = None
        for el in contributions:
            if self.aggregration_method == 'mean':
                aggregated_element = sum(el) / len(el)
            elif self.aggregration_method == 'first':
                aggregated_element = el[0]
            elif self.aggregration_method == 'sum':
                aggregated_element = sum(el)
            elif self.aggregration_method == 'max':
                aggregated_element = max(el)
            aggregated_contributions.append(aggregated_element)
        return aggregated_contributions

    def _aggregate_subtokens_contributions(self, contribs, alignment_ids, normalize=False):
        contribs = contribs[1:-1]
        agg_contribs = [[] for _ in range(len(set(alignment_ids)))]
        # if normalize:
        #     contribs = normalize_contribs(contribs)
        for al_id, contrib in zip(alignment_ids, contribs):
            agg_contribs[al_id].append(contrib)
        agg_contribs = self._compute_aggregation(agg_contribs)
        return agg_contribs

    def get_contributions(self, subwords_alignment_dict):
        sentences_contribs = dict()
        for sent_id in tqdm(subwords_alignment_dict):
            model_input = subwords_alignment_dict[sent_id]['model_input']
            al_ids = subwords_alignment_dict[sent_id]['alignment_ids']
            contribs = self.compute_sentence_contributions(model_input)
            # cls_contribs = contribs[self.layer][0].tolist()
            cls_contribs = contribs[0].tolist()
            agg_contribs = self._aggregate_subtokens_contributions(cls_contribs, al_ids)
            sentences_contribs[sent_id] = agg_contribs

        return sentences_contribs


class ValueZeroingContributionExtractor(TokenContributionExtractor, ABC):

    def _load_model(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        if 'xlm' in model_name:
            if not self.random_init:
                model = XLMRobertaForMaskedLMVZ.from_pretrained(model_name, config=config)
            else:
                model = XLMRobertaForMaskedLMVZ._from_config(config)
        elif 'roberta' in model_name:
            if not self.random_init:
                model = RobertaForMaskedLMVZ.from_pretrained(model_name, config=config)
            else:
                model = RobertaForMaskedLMVZ._from_config(config)
        elif 'camembert' or 'gilberto' in model_name:
            if not self.random_init:
                model = CamembertForMaskedLMVZ.from_pretrained(model_name, config=config)
            else:
                model = CamembertForMaskedLMVZ._from_config(config)
        else:
             model = None
        model.to(self.device)
        return model

    def _compute_joint_attention(self, att_mat, res=True):
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

    def compute_sentence_contributions(self, tokenized_text):
        tokenized_text = {k: v.to(self.device) for k, v in tokenized_text.items()}

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
            for t in range(seq_length):
                try:
                    extended_blanking_attention_mask: torch.Tensor = self.model.bert.get_extended_attention_mask(
                        tokenized_text['attention_mask'], input_shape)  # , device=self.device)
                except:
                    extended_blanking_attention_mask: torch.Tensor = self.model.roberta.get_extended_attention_mask(
                        tokenized_text['attention_mask'], input_shape)  # , device=self.device)

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
            layer_valuezeroing_scores = valuezeroing_scores[self.layer]
            return layer_valuezeroing_scores

        rollout_valuezeroing_scores = self._compute_joint_attention(valuezeroing_scores, res=False)
        layer_rollout_valuezeroing_scores = rollout_valuezeroing_scores[self.layer]
        return layer_rollout_valuezeroing_scores


class AltiContributionExtractor(TokenContributionExtractor, ABC):

    def _load_model(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        if not self.random_init:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            model = AutoModelForMaskedLM.from_config(config)
        model.to(self.device)
        return ModelWrapper(model)

    def compute_sentence_contributions(self, tokenized_text):
        tokenized_text.to(self.device)
        prediction_scores, hidden_states, attentions, contributions_data = self.model(tokenized_text)
        resultant_norm = torch.norm(torch.squeeze(contributions_data['resultants']), p=1, dim=-1)
        normalized_contributions = normalize_contributions(contributions_data['contributions'], scaling='min_sum',
                                                           resultant_norm=resultant_norm)
        contributions_mix = compute_joint_attention(normalized_contributions)
        contributions_mix = contributions_mix.detach().cpu().numpy()
        layer_contributions_mix = contributions_mix[self.layer]
        return layer_contributions_mix


class AttentionMatrixExtractor(TokenContributionExtractor, ABC):

    def _load_model(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        if not self.random_init:
            model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True)
        else:
            config.output_attentions = True
            model = AutoModelForMaskedLM.from_config(config) #, output_attentions=True)
        model.to(self.device)
        return model

    def compute_sentence_contributions(self, tokenized_text):
        tokenized_text.to(self.device)
        model_output = self.model(**tokenized_text)
        layer_attention_matrix = model_output['attentions'][self.layer]
        layer_attention_matrix = layer_attention_matrix.detach().squeeze()
        avg_attention_matrix = torch.mean(layer_attention_matrix, dim=0)
        return avg_attention_matrix


class EyeTrackingDataLoader:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def __load_and_merge_users_dfs(self) -> pd.DataFrame:
        users_dfs = []
        for user_file_name in os.listdir(self.data_dir):
            src_path = os.path.join(self.data_dir, user_file_name)
            user_df = pd.read_csv(src_path)[['trialid', 'sentnum', 'ianum', 'ia']]
            users_dfs.append(user_df)
        merged_df = pd.concat(users_dfs, ignore_index=True).drop_duplicates()
        return merged_df

    def load_sentences(self) -> pd.DataFrame:
        """
        This method creates a DataFrame with the following columns:
        - sent_id: containing a unique key for the sentence computed as the concatenation of 'trialid' and 'sentnum'
        - sentence: contains the list of words of a sentence, sorted by 'ianum'
        A bit of processing is necessary since the sentences have been split in pieces randomly located on users files.
        Moreover, not all users read all the sentences, and all pieces of it.
        """
        merged_user_df = self.__load_and_merge_users_dfs()
        sentences_df = merged_user_df.sort_values(by=['trialid', 'sentnum', 'ianum']).groupby(['trialid', 'sentnum'])[
            'ia'].apply(list).reset_index()
        sent_id_column = sentences_df['trialid'].astype(int).astype(str) + '_' + sentences_df['sentnum'].astype(
            int).astype(str)
        sentences_df.insert(0, 'sent_id', sent_id_column)  # I wanted it in first position :)
        sentences_df = sentences_df.drop(['trialid', 'sentnum'], axis=1)
        sentences_df.rename(columns={'ia': 'sentence'}, inplace=True)
        return sentences_df


def align_to_original_words(model_tokens: list, original_tokens: list, subword_prefix: str,
                            lowercase: bool = False) -> list:
    if lowercase:
        original_tokens = [tok.lower() for tok in original_tokens]
    model_tokens = model_tokens[1: -1]  # Remove <s> and </s>
    aligned_model_tokens = []
    alignment_ids = []
    alignment_id = -1
    orig_idx = 0
    for token in model_tokens:
        alignment_id += 1
        if token.startswith(subword_prefix):  # Remove the sub-word prefix
            token = token[len(subword_prefix):]
        if len(aligned_model_tokens) == 0:  # First token (serve?)
            aligned_model_tokens.append(token)
        elif aligned_model_tokens[-1] + token in original_tokens[
            orig_idx]:  # We are in the second (third, fourth, ...) sub-token
            aligned_model_tokens[-1] += token  # so we merge the token with its preceding(s)
            alignment_id -= 1
        else:
            aligned_model_tokens.append(token)
        if aligned_model_tokens[-1] == original_tokens[
            orig_idx]:  # A token was equal to an entire original word or a set of
            orig_idx += 1  # sub-tokens was merged and matched an original word
        alignment_ids.append(alignment_id)

    if aligned_model_tokens != original_tokens:
        raise Exception(
            f'Failed to align tokens.\nOriginal tokens: {original_tokens}\nObtained alignment: {aligned_model_tokens}')
    return alignment_ids


def create_subwords_alignment(dataset: Dataset, tokenizer: AutoTokenizer, subword_prefix: str,
                              lowercase: bool = False) -> dict:
    sentence_alignment_dict = dict()

    for sent_id, sentence in zip(dataset['id'], dataset['text']):
        for tok_id, tok in enumerate(sentence):
            if tok == '–':
                sentence[tok_id] = '-'
        if lowercase:
            sentence = [word.lower() for word in sentence]
        tokenized_sentence = tokenizer(sentence, is_split_into_words=True, return_tensors='pt')
        input_ids = tokenized_sentence['input_ids'].tolist()[0]  # 0 because the batch_size is 1
        model_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        alignment_ids = align_to_original_words(model_tokens, sentence, subword_prefix, lowercase)
        sentence_alignment_dict[sent_id] = {'model_input': tokenized_sentence, 'alignment_ids': alignment_ids}
    return sentence_alignment_dict


def save_dictionary(dictionary, out_path):
    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(dictionary))


def get_model_subword_prefix(tokenizer_name):
    if tokenizer_name == 'xlm-roberta-base' or tokenizer_name == 'idb-ita/gilberto-uncased-from-camembert':
        return '▁'
    elif tokenizer_name == 'roberta-base':
        return 'Ġ'
    else:
        return None