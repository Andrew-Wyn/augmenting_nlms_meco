import os
import sys

sys.path.append(os.path.abspath("."))  #  run the scrpits file from the parent folder

from anm.gaze_dataloader.dataset import _create_senteces_from_data
from anm.attn_correlation.utils import *
from transformers import AutoTokenizer
import pandas as pd
import argparse
import torch


def get_model_path(language_mode, training_mode, language, finetuned_models_dir, user_id=None):
    # if 'not_finetuned' in training_mode:
    #     if language_mode == 'cross_lingual':
    #         return 'xlm-roberta-base'
    #     elif language == 'it':
    #         return 'idb-ita/gilberto-uncased-from-camembert'
    #     else:
    #         return 'roberta-base'
    # else:
    pretrained = 'np' if 'not_pretrained' in training_mode else 'p'
    finetuned =  'nf' if 'not_finetuned' in training_mode else f'f_{language}{user_id}' 
    if language_mode == 'cross_lingual':
        if finetuned == 'nf':
            model_string = f'xlm_{language}'
        else:
            model_string = 'xlm'
    elif language == 'it':
        model_string = 'camem'
    else:
        model_string = 'roberta'
    # model_dir = f'{finetuned_models_dir}/gaze_finetuning_{language}_{user_id}_{pretrained}_{model_string}'
    model_dir = f'{model_string}_{pretrained}_{finetuned}'
    model_path = os.path.join(finetuned_models_dir, model_dir)
    # for file_name in os.listdir(model_dir):
    #     file_path = os.path.join(model_dir, file_name)
    #     if file_name != 'tf_logs' and os.path.isdir(file_path):
    #         if 'config.json' in os.listdir(file_path):
    #             model_path = file_path
    #         else:
    #             inner_dir = os.listdir(file_path)[0]
    #             model_path = os.path.join(file_path, inner_dir)
    return model_path

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

def get_languages(task):
    if task == 'sentipolc':
        return ['it']
    elif task == 'sst2':
        return ['en']
    else:
        return ['it', 'en']


def get_src_dir(task):
    if task == 'sst2':
        return '/home/lmoroni/__workdir/augmenting_nlms_meco/output/sst2'
    elif task == 'complexity':
        return '/home/lmoroni/__workdir/augmenting_nlms_meco/output/complexity'
    elif task == 'complexity_binary':
        return '/home/lmoroni/__workdir/augmenting_nlms_meco/output/complexity_binary'
    elif task == 'sentipolc':
        return '/home/luca/Workspace/augmenting_nlms_meco/output/sentipolc'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', choices=['valuezeroing', 'alti', 'attention', 'valuezeroing_rollout'])
    parser.add_argument('-t', '--task', choices=['sst2', 'sentipolc', 'complexity', 'complexity_binary'])
    args = parser.parse_args()

    rollout = True if args.method == 'valuezeroing_rollout' else False

    language_modes = ['mono_lingual', 'cross_lingual']
    training_modes = ['not_pretrained_not_finetuned']#['not_pretrained_finetuned', 'not_pretrained_not_finetuned', 'pretrained_not_finetuned', 'pretrained_finetuned']
    languages = get_languages(args.task)

    
    eye_tracking_data_dir = 'augmenting_nlms_meco_data/'
    attn_results_dir = f'output/attn_data/{args.task}'
    finetuned_models_dir = get_src_dir(args.task)

    if not os.path.exists(attn_results_dir):
        os.mkdir(attn_results_dir)

    print(f'METHOD: {args.method}')
    for language in languages:
        print(f'LANGUAGE: {language}')
        language_src_dir = os.path.join(eye_tracking_data_dir, language)
        for file_name in [file_name for file_name in os.listdir(language_src_dir) if
                          file_name not in ['.ipynb_checkpoints', 'all_mean_dataset.csv']]:
            user_id = file_name.split('_')[1]
            data = pd.read_csv(os.path.join(language_src_dir, file_name), index_col=0)
            gaze_dataset = _create_senteces_from_data(data, [], keep_id=True)
            for language_mode in language_modes:
                lowercase = True if (language == 'it' and language_mode == 'mono_lingual') else False
                print(f'LANGUAGE MODE: {language_mode}')
                for training_mode in training_modes:
                    print(f'TRAINING MODE: {training_mode}')
                    random_init = False
                    out_dir = os.path.join(attn_results_dir, args.method, language, language_mode, training_mode,
                                           user_id)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    model_path = get_model_path(language_mode, training_mode, language, finetuned_models_dir, user_id)
                    print(f'MODEL PATH: {model_path}')
                    print(f'OUT DIR: {out_dir}')
                    print('____________________________________')
                    for layer in range(12):
                        out_path = os.path.join(out_dir, f'{layer}.json')
                        #if not os.path.exists(out_path):
                        random_init = False
                        extract_attention(args.method, gaze_dataset, model_path, out_path, layer, rollout, lowercase, random_init)




if __name__ == '__main__':
    main()
