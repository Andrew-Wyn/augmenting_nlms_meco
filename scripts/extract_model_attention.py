import os
import sys

sys.path.append(os.path.abspath("."))  #  run the scrpits file from the parent folder

from anm.gaze_dataloader.dataset import _create_senteces_from_data
from anm.attn_correlation.utils import *
from transformers import AutoTokenizer
import pandas as pd
import argparse


def get_model_path(language_mode, training_mode, language, finetuned_models_dir, user_id=None):
    if training_mode == 'pretrained_not_finetuned':
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
        print('valuezeroing', rollout)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', choices=['valuezeroing', 'alti', 'attention', 'valuezeroing_rollout', 'dig'])
    args = parser.parse_args()

    rollout = True if args.method == 'valuezeroing_rollout' else False

    language_modes = ['mono_lingual', 'cross_lingual']
    training_modes = ['pretrained_not_finetuned', 'pretrained_finetuned', 'not_pretrained_finetuned', 'not_pretrained_not_finetuned']
    languages = ['en', 'it']

    eye_tracking_data_dir = 'augmenting_nlms_meco_data/'
    finetuned_models_dir = '/home/lmoroni/__workdir/augmenting_nlms_meco/output'
    attn_results_dir = 'output/attn_data/base/'

    print(f'METHOD: {args.method}')
    for language in languages:
        print(f'LANGUAGE: {language}')
        language_src_dir = os.path.join(eye_tracking_data_dir, language)
        for file_name in [file_name for file_name in os.listdir(language_src_dir) if
                          file_name not in ['all_mean_dataset.csv', '.ipynb_checkpoints']]:
            user_id = file_name.split('_')[1]
            data = pd.read_csv(os.path.join(language_src_dir, file_name), index_col=0)
            gaze_dataset = _create_senteces_from_data(data, [], keep_id=True)
            for language_mode in language_modes:
                lowercase = True if (language == 'it' and language_mode == 'mono_lingual') else False
                print(f'LANGUAGE MODE: {language_mode}')
                for training_mode in training_modes:
                    print(f'TRAINING MODE: {training_mode}')
                    random_init = True if training_mode == 'not_pretrained_not_finetuned' else False
                    out_dir = os.path.join(attn_results_dir, args.method, language, language_mode, training_mode,
                                           user_id)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    model_path = get_model_path(language_mode, training_mode, language, finetuned_models_dir, user_id)
                    for layer in range(12):
                        out_path = os.path.join(out_dir, f'{layer}.json')
                        if not os.path.exists(out_path):
                            extract_attention(args.method, gaze_dataset, model_path, out_path, layer, rollout, lowercase, random_init)






if __name__ == '__main__':
    main()
