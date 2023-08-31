from utils import EyeTrackingDataLoader, ValueZeroingContributionExtractor
from transformers import AutoTokenizer
import pandas as pd
import argparse
import json
import os


def align_to_original_words(model_tokens: list, original_tokens: list, subword_prefix: str) -> list:
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


def create_subwords_alignment(sentences_df: pd.DataFrame, tokenizer: AutoTokenizer, subword_prefix: str) -> dict:
    sentence_alignment_dict = dict()

    for idx, row in sentences_df.iterrows():
        sent_id = row['sent_id']
        sentence = row['sentence']
        for tok_id, tok in enumerate(sentence):
            if tok == '–':
                sentence[tok_id] = '-'
        tokenized_sentence = tokenizer(sentence, is_split_into_words=True, return_tensors='pt')
        input_ids = tokenized_sentence['input_ids'].tolist()[0]  # 0 because the batch_size is 1
        model_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        alignment_ids = align_to_original_words(model_tokens, sentence, subword_prefix)
        sentence_alignment_dict[sent_id] = {'model_input': tokenized_sentence, 'alignment_ids': alignment_ids}
    return sentence_alignment_dict


def save_dictionary(dictionary, out_path):
    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(dictionary))


def get_model_subword_prefix(model_name):
    if model_name == 'xlm-roberta-base':
        return '▁'
    elif model_name == 'roberta-base':
        return 'Ġ'
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-r', '--rollout', action='store_true')
    parser.add_argument('-a', '--aggregation_method', choices=['max', 'first', 'mean', 'sum'])
    parser.add_argument('-l', '--layer', default=-1, type=int)
    parser.add_argument('-n', '--language', default='en')
    args = parser.parse_args()

    print(f'Aggregation mehtod = {args.aggregation_method}')
    print(f'Rollout = {args.rollout}')

    eye_tracking_data_dir = f'../augmenting_nlms_meco_data/{args.language}'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    subword_prefix = get_model_subword_prefix(args.model_name)

    out_dir = f'data/value_zeroing/{args.language}/{args.model_name}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    file_name = f'{args.aggregation_method}' if not args.rollout else f'{args.aggregation_method}_rollout'
    file_name += f'_l{args.layer}.json'
    out_path = os.path.join(out_dir, file_name)

    dl = EyeTrackingDataLoader(eye_tracking_data_dir)
    sentences_df = dl.load_sentences()

    sentence_alignment_dict = create_subwords_alignment(sentences_df, tokenizer, subword_prefix)

    attn_extractor = ValueZeroingContributionExtractor(args.model_name, args.layer, args.rollout,
                                                       args.aggregation_method, 'cpu')
    sentences_contribs = attn_extractor.get_contributions(sentence_alignment_dict)

    save_dictionary(sentences_contribs, out_path)


if __name__ == '__main__':
    main()
