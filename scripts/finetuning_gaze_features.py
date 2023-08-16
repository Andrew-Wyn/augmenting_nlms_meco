import os
import sys
sys.path.append(os.path.abspath(".")) # run the scrpits file from the parent folder

# from anm.modeling.multitask_camembert import CamembertForMultiTaskTokenClassification
from anm.modeling.multitask_roberta import RobertaForMultiTaskTokenClassification
from anm.gaze_dataloader.datacollator import DataCollatorForMultiTaskTokenClassification
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from anm.gaze_dataloader.dataset import _create_senteces_from_data, create_tokenize_and_align_labels_map

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

model_name = 'roberta-base'
model = RobertaForMultiTaskTokenClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

data = pd.read_csv("augmenting_nlms_meco_data/en/en_6_dataset.csv", index_col=0)
gaze_dataset = _create_senteces_from_data(data)

features = [col_name for col_name in gaze_dataset.column_names if col_name.startswith('label_')]

tokenized_dataset = gaze_dataset.map(
            create_tokenize_and_align_labels_map(tokenizer, features),
            batched=True,
            remove_columns=gaze_dataset.column_names,
            # desc="Running tokenizer on dataset",
)

print('tokens: ', tokenizer.convert_ids_to_tokens(tokenized_dataset[0]['input_ids']))
for key in tokenized_dataset[0].keys():
    print(f'{key}:', tokenized_dataset[0][key])

data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
dataloader = DataLoader(tokenized_dataset, shuffle=True, collate_fn=data_collator, batch_size=2)

for step, batch in enumerate(dataloader):
    print(batch)
    outputs = model(**batch)
    print(outputs)
    break