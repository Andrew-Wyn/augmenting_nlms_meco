from transformers import get_linear_schedule_with_warmup
import torch
import logging.config
import torch.nn.functional as F
import json
from sklearn.preprocessing import MinMaxScaler
from anm.modeling.multitask_roberta import RobertaForMultiTaskTokenClassification
from anm.modeling.multitask_xlm_roberta import XLMRobertaForMultiTaskTokenClassification
from anm.modeling.multitask_camembert import CamembertForMultiTaskTokenClassification
import numpy as np
from transformers import (
    AutoConfig
)

CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "processing": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}


logging.config.dictConfig(CONFIG)
LOGGER = logging.getLogger("processing")


class Config:
    @classmethod
    def load_json(cls, fpath):
        cf = cls()
        with open(fpath) as f:
            cf.__dict__ = json.load(f)

        return cf


def create_scheduler(cf, optim, dl):
    """
    Creates a linear learning rate scheduler.
    """
    n_iters = cf.n_epochs * len(dl)
    return get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=n_iters)


def load_model_from_hf(model_type, model_name, pretrained):
    cf = Config.load_json("configs/modeling_configuration.json")
    config = AutoConfig.from_pretrained(model_name)
    config.update({"tasks": cf.tasks})

    model = None
    # Model
    LOGGER.info("Initiating model ...")

    if not pretrained:
        # initiate model with random weights
        # You can initialize a random BERT model using the Hugginface capabilites 
        # (from the documentation https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/bert#transformers.BertConfig)
        LOGGER.info("Take randomized model")

        if model_type == "Roberta":
            model = RobertaForMultiTaskTokenClassification(config)
        elif model_type == "XLM":
            model = XLMRobertaForMultiTaskTokenClassification(config)
        elif model_type == "Camembert":
            model = CamembertForMultiTaskTokenClassification(config)
    else:
        LOGGER.info("Take pretrained model")

        if model_type == "Roberta":
            model = RobertaForMultiTaskTokenClassification.from_pretrained(model_name, config=config)
        elif model_type == "XLM":
            model = XLMRobertaForMultiTaskTokenClassification.from_pretrained(model_name, config=config)
        elif model_type == "Camembert":
            model = CamembertForMultiTaskTokenClassification.from_pretrained(model_name, config=config)

    return model


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
    out_dir_1 = 'output/attn_data/users'
    out_dir_2 = os.path.join(out_dir_1 , model_string)
    out_dir_3 = os.path.join(out_dir_2, method)
    out_path = os.path.join(out_dir_3, f'{layer}.json' if not rollout else f'{layer}_rollout.json' )
    for directory in [out_dir_1, out_dir_2, out_dir_3]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    return out_path