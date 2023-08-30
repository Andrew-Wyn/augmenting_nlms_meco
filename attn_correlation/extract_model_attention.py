from utils.model_attention_utils import *
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import json
import os


def align_to_human_tokens(model_tokens, human_tokens, split_chars):
    model_tokens = model_tokens[1: -1]
    aligned_model_tokens = []
    alignment_ids = []
    alignment_id = -1
    h_idx = 0
    for token in model_tokens:
        alignment_id += 1
        if token.startswith(split_chars):
            token = token[len(split_chars):]
        if len(aligned_model_tokens) == 0:
            aligned_model_tokens.append(token)
        elif aligned_model_tokens[-1] + token in human_tokens[h_idx]:
            aligned_model_tokens[-1] += token
            alignment_id -= 1
        else:
            aligned_model_tokens.append(token)
        if aligned_model_tokens[-1] == human_tokens[h_idx]:
            h_idx += 1
        alignment_ids.append(alignment_id)

    return aligned_model_tokens, alignment_ids


def load_sentences(src_dir):
    sentences_dict = dict()
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        for line in open(src_path):
            line = line.strip().split('\t')
            sentences_dict[int(line[0])] = line[1]
    return sentences_dict


def check_alignment(model_tokens, human_tokens):
    if len(model_tokens) != len(human_tokens):
        return False
    for idx in range(len(model_tokens)):
        if model_tokens[idx] != human_tokens[idx]:
            return False
    return True


def create_subwords_alignment(sentences_dict, tokenizer, split_chars):
    model_sent_dict = dict()

    for sent_id in sentences_dict.keys():
        human_tokens = sentences_dict[sent_id].split(' ')
        tokenized_sentence = tokenizer(human_tokens, is_split_into_words=True, return_tensors='pt')
        model_tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence['input_ids'].tolist()[0])
        aligned_tokens, alignment_ids = align_to_human_tokens(model_tokens, human_tokens, split_chars)
        model_sent_dict[sent_id] = {'model_input': tokenized_sentence, 'alignment_ids': alignment_ids}
        if not check_alignment(aligned_tokens, human_tokens):
            print(model_tokens)
            print(human_tokens)
            assert False

    return model_sent_dict


def compute_sentence_contributions(value_zeroing, subwords_alignment_dict, layer, agg_method):
    sentences_contribs = dict()
    for sent_id in tqdm(subwords_alignment_dict):
        model_input = subwords_alignment_dict[sent_id]['model_input']
        al_ids = subwords_alignment_dict[sent_id]['alignment_ids']
        contribs = value_zeroing.get_contributions(model_input)
        cls_contribs = contribs[layer][0].tolist()
        agg_contribs = aggregate_subtokens_contribs(cls_contribs, al_ids, agg_method=agg_method)
        sentences_contribs[sent_id] = agg_contribs

    return sentences_contribs


def save_dictionary(dictionary, out_path):
    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(dictionary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rollout', action='store_true')
    parser.add_argument('-a', '--aggregation_method', choices=['max', 'first', 'mean', 'sum'])
    parser.add_argument('-l', '--layer', default=-1, type=int)
    parser.add_argument('-n', '--language', default='en')
    args = parser.parse_args()

    print(f'Aggregation mehtod = {args.aggregation_method}')
    print(f'Rollout = {args.rollout}')

    sentences_dir = f'data/meco_l1/sentences/{args.language}'

    # model_name = 'bert-base-uncased'
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    split_chars = '▁'#'##'

    out_dir = f'data/meco_l1/value_zeroing/{args.language}/roberta'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    file_name = f'{args.aggregation_method}' if not args.rollout else f'{args.aggregation_method}_rollout'
    file_name += f'_l{args.layer}.json'
    out_path = os.path.join(out_dir, file_name)

    sentences_dict = load_sentences(sentences_dir)
    subwords_alignment_dict = create_subwords_alignment(sentences_dict, tokenizer, split_chars)
    value_zeroing = ValueZeroingContributions(model_name, rollout=args.rollout, layer=args.layer)
    sentences_contribs = compute_sentence_contributions(value_zeroing, subwords_alignment_dict, args.layer,
                                                        args.aggregation_method)

    save_dictionary(sentences_contribs, out_path)


if __name__ == '__main__':
    main()
