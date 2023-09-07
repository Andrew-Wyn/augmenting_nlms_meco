from utils import (
    EyeTrackingDataLoader,
    AttentionMatrixExtractor,
    AltiContributionExtractor,
    ValueZeroingContributionExtractor
)
from transformers import AutoTokenizer
import pandas as pd
import argparse
import torch
import json
import os


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


def create_subwords_alignment(sentences_df: pd.DataFrame, tokenizer: AutoTokenizer, subword_prefix: str,
                              lowercase: bool = False) -> dict:
    sentence_alignment_dict = dict()

    for idx, row in sentences_df.iterrows():
        sent_id = row['sent_id']
        sentence = row['sentence']
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


def get_tokenizer_name(model_name):
    if model_name == 'xlm':
        return 'xlm-roberta-base'
    elif model_name == 'roberta':
        return 'roberta-base'
    elif model_name == 'camem':
        return 'idb-ita/gilberto-uncased-from-camembert'
    else:
        return None
    

def extract_attention(method, model_name, out_path, src_model_path, language, layer, rollout, lowercase):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eye_tracking_data_dir = f'../../augmenting_nlms_meco_data/{language}'


    tokenizer_name = get_tokenizer_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    subword_prefix = get_model_subword_prefix(tokenizer_name)


    dl = EyeTrackingDataLoader(eye_tracking_data_dir)
    sentences_df = dl.load_sentences()

    sentence_alignment_dict = create_subwords_alignment(sentences_df, tokenizer, subword_prefix, lowercase)

    if method == 'valuezeroing':
        attn_extractor = ValueZeroingContributionExtractor(src_model_path, layer, rollout, 'first', device, tokenizer_name)
    elif method == 'alti':
        attn_extractor = AltiContributionExtractor(src_model_path, layer, rollout, 'first', device)
    else:
        attn_extractor = AttentionMatrixExtractor(src_model_path, layer, rollout, 'first', device)

    sentences_contribs = attn_extractor.get_contributions(sentence_alignment_dict)

    save_dictionary(sentences_contribs, out_path)

def get_and_create_out_path(model_string, method, layer, rollout):
    out_dir_1 = 'data/users'
    out_dir_2 = os.path.join(out_dir_1 , model_string)
    out_dir_3 = os.path.join(out_dir_2, method)
    out_path = os.path.join(out_dir_3, f'{layer}.json' if not rollout else f'{layer}_rollout.json' )
    for directory in [out_dir_1, out_dir_2, out_dir_3]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--method', choices=['valuezeroing', 'alti', 'attention'])
    parser.add_argument('-r', '--rollout', action='store_true')
    args = parser.parse_args()
    
    models_names = ['xlm', 'roberta', 'camem']
    languages = ['en', 'it']
    layers = list(range(12))
    en_userids = [49, 57, 6, 83, 98]
    it_userids = [1, 26, 38, 43, 44]

    finetuned_models_dir = '/home/lmoroni/__workdir/augmenting_nlms_meco/output'

    for language in languages:
        userids = it_userids if language == 'it' else en_userids
        for user_id in userids:
            for pretrained_mode in ['p', 'np']:
                for model_name in models_names:
                    lowercase = True if model_name == 'camem' else False
                    src_model_dir = os.path.join(finetuned_models_dir, f'gaze_finetuning_{language}_{user_id}_{pretrained_mode}_{model_name}')
                    if not os.path.exists(src_model_dir):
                        continue
                    for file_name in os.listdir(src_model_dir):
                        file_path = os.path.join(src_model_dir, file_name)
                        if file_name != 'tf_logs' and os.path.isdir(file_path):
                            if 'config.json' in os.listdir(file_path):
                                src_model_path = file_path
                            else:
                                inner_dir = os.listdir(file_path)[0]
                                src_model_path = os.path.join(file_path, inner_dir)
                       
                    print(f'Metod = {args.method}, rollout = {args.rollout}')
                    print(f'Model = {model_name}')
                    print(f'User = {user_id}')
                    print(f'Pretrained = {pretrained_mode}')
                    
                    for layer in layers:
                        print(f'Layer = {layer}')
                        model_string = f'{language}_{user_id}_{pretrained_mode}_{model_name}'
                        out_path = get_and_create_out_path(model_string, args.method, layer, args.rollout)
                        if not os.path.exists(out_path):
                            extract_attention(args.method, model_name, out_path, src_model_path, language, layer, args.rollout, lowercase)
                            print('\n\n')
                    print('\n____________________________________________________________________\n')
    


if __name__ == '__main__':
    main()
