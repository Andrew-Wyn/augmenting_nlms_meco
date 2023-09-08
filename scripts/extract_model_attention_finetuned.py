import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

from anm.attn_correlation.utils import *
from transformers import AutoTokenizer
import pandas as pd
import argparse
import torch
import json
import os


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
