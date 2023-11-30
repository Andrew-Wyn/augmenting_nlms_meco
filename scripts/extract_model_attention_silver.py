import os
import sys

sys.path.append(os.path.abspath("."))  #  run the scrpits file from the parent folder

from anm.gaze_dataloader.dataset import _create_senteces_from_data
from anm.attn_correlation.utils import *
from transformers import AutoTokenizer
import pandas as pd
import argparse
import torch

def get_tokenizer_name(model_name):
    if 'xlm' in model_name:
        return 'xlm-roberta-base'
    elif 'roberta' in model_name:
        return 'roberta-base'
    else:
        return 'idb-ita/gilberto-uncased-from-camembert'


def extract_attention(method, gaze_dataset, model_name, out_path, layer, rollout, lowercase, random_init):
    tokenizer_name = get_tokenizer_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    subword_prefix = get_model_subword_prefix(tokenizer_name)

    sentence_alignment_dict = create_subwords_alignment(gaze_dataset, tokenizer, subword_prefix, lowercase)

    if 'valuezeroing' in method:
        attn_extractor = ValueZeroingContributionExtractor(model_name, layer, rollout, 'first', random_init)
    elif method == 'alti':
        attn_extractor = AltiContributionExtractor(model_name, layer, rollout, 'first', random_init)
    elif method == 'dig':
        attn_extractor = DIGAttrExtractor(tokenizer,
                                          model_name, 
                                          layer, 
                                          rollout, 
                                          'first', 
                                          random_init)
    else:
        attn_extractor = AttentionMatrixExtractor(model_name, layer, rollout, 'first', random_init)

    sentences_contribs = attn_extractor.get_contributions(sentence_alignment_dict)
    save_dictionary(sentences_contribs, out_path)


def get_all_silver_data(silver_data_src_dir, task, language, language_mode, only_pt=True):
    model_str = None
    if language_mode == 'cross_lingual':
        model_str = 'xlm'
    elif language == 'en':
        model_str = 'roberta'
    else:
        model_str = 'camem'
    dir_name = f'silver_{task}_{language}_{model_str}'
    config_dir = os.path.join(silver_data_src_dir, dir_name)
    file_names = os.listdir(config_dir)
    if only_pt:
        file_names = [file_name for file_name in file_names if file_name.startswith('p')]
    return [os.path.join(silver_data_src_dir, config_dir, file_name) for file_name in file_names]

def get_languages(task):
    if task == 'sentipolc':
        return ['it']
    elif task == 'sst2':
        return ['en']
    else:
        return ['it', 'en']



def get_model_path(language_mode, training_mode, language, finetuned_models_dir, user_id=None):
    if 'not_finetuned' in training_mode:
        if language_mode == 'cross_lingual':
            return 'xlm-roberta-base'
        elif language == 'it':
            return 'idb-ita/gilberto-uncased-from-camembert'
        else:
            return 'roberta-base'
    else:
        pretrained = 'p' if training_mode == 'pretrained_finetuned' else 'np'
        if language_mode == 'cross_lingual':
            model_string = 'xlm'
        elif language == 'it':
            model_string = 'camem'
        else:
            model_string = 'roberta'
        model_dir = f'{finetuned_models_dir}/gaze_finetuning_{language}_{user_id}_{pretrained}_{model_string}'
        for file_name in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file_name)
            if file_name != 'tf_logs' and os.path.isdir(file_path):
                if 'config.json' in os.listdir(file_path):
                    model_path = file_path
                else:
                    inner_dir = os.listdir(file_path)[0]
                    model_path = os.path.join(file_path, inner_dir)
        return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', choices=['valuezeroing', 'alti', 'attention', 'valuezeroing_rollout', 'dig'])
    parser.add_argument('-t', '--task', choices=['sst2', 'sentipolc', 'complexity', 'complexity_binary'])
    args = parser.parse_args()
    
    languages = get_languages(args.task)
    
    language_modes = ['mono_lingual', 'cross_lingual']
    training_modes = ['not_pretrained_finetuned', 'not_pretrained_not_finetuned', 'pretrained_not_finetuned', 'pretrained_finetuned']
    
    finetuned_models_dir = '/home/lmoroni/__workdir/augmenting_nlms_meco/output'
    silver_data_src_dir = '/home/lmoroni/__workdir/augmenting_nlms_meco/augmenting_nlms_meco_data/'
    attn_results_dir = f'output/attn_data_silver/base/{args.task}'
    
    if not os.path.exists(attn_results_dir):
        os.mkdir(attn_results_dir)
    
    rollout = False
    
    for language in ['en']:# languages:
        print(f'LANGUAGE: {language}')
        for language_mode in language_modes:
            print(f'LANGUAGE MODE: {language_mode}')
            lowercase = True if (language == 'it' and language_mode == 'mono_lingual') else False
            silver_data_paths = get_all_silver_data(silver_data_src_dir, args.task, language, language_mode, only_pt=True)
            for silver_data_path in silver_data_paths:
                user_id = silver_data_path.split('_')[-1].split('.')[0]
                data = pd.read_csv(silver_data_path, index_col=0)
                gaze_dataset = _create_senteces_from_data(data, [], keep_id=True)
                print(f'USER ID: {user_id}')
                for training_mode in training_modes:
                    print(f'TRAINING MODE: {training_mode}')
                    model_path = get_model_path(language_mode, training_mode, language, finetuned_models_dir, user_id=user_id)
                    random_init = True if training_mode == 'not_pretrained_not_finetuned' else False
                    out_dir = os.path.join(attn_results_dir, args.method, language, language_mode, training_mode, user_id)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    layers = range(12) if args.method != 'dig' else [0]
                    for layer in layers:
                        out_path = os.path.join(out_dir, f'{layer}.json')
                        if not os.path.exists(out_path):
                            contribs = extract_attention(args.method, gaze_dataset, model_path, out_path, layer, rollout, lowercase, random_init)


if __name__ == '__main__':
    main()