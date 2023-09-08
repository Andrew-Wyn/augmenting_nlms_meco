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
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-e', '--method', choices=['valuezeroing', 'alti', 'attention'])
    parser.add_argument('-r', '--rollout', action='store_true')
    parser.add_argument('-a', '--aggregation_method', choices=['max', 'first', 'mean', 'sum'])
    parser.add_argument('-l', '--layer', default=-1, type=int)
    parser.add_argument('-n', '--language', default='en')
    parser.add_argument('-o', '--lowercase', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print(f'Attention extraction method = {args.method}')
    print(f'Aggregation mehtod = {args.aggregation_method}')
    print(f'Rollout = {args.rollout}')

    eye_tracking_data_dir = f'../../augmenting_nlms_meco_data/{args.language}'

    model_name_for_dir = get_model_name_for_directory(args.model_name)
    out_dir = f'output/attn_data/{args.method}/{args.language}/{model_name_for_dir}'
    for directory in [f'output/attn_data/{args.method}', f'output/attn_data/{args.method}/{args.language}', out_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    file_name = f'{args.aggregation_method}' if not args.rollout else f'{args.aggregation_method}_rollout'
    file_name += f'_l{args.layer}.json'
    out_path = os.path.join(out_dir, file_name)

    extract_attention(args.method, args.model_name, out_path, args.model_name, args.language, args.layer, args.rollout, args.lowercase)

if __name__ == '__main__':
    main()
