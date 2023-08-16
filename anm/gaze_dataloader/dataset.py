import numpy as np
from datasets import Dataset


def _create_senteces_from_data(data):
    dropping_cols = {"sentnum", "ia", "lang", "trialid", "ianum", "uniform_id", "sentence_id"}
    
    # sort by sentnum and ianum, to avoid splitted sentences
    data = data.sort_values(by=["sentnum", "ianum"])

    # create sentence_id
    data["sentence_id"] = data["trialid"].astype(int).astype(str) + data["sentnum"].astype(int).astype(str)

    labels = [l for l in data.columns if l not in dropping_cols]

    word_func = lambda s: [w for w in s["ia"].values.tolist()]

    features_func = lambda s: [np.array(s.drop(columns=dropping_cols).iloc[i])
                            for i in range(len(s))]

    sentences = data.groupby("sentence_id").apply(word_func).tolist()

    targets = data.groupby("sentence_id").apply(features_func).tolist()

    data_list = []

    for s, t in zip(sentences, targets):
        data_list.append({
            **{"text": s,},
            **{"label_"+str("-".join(l.split("."))) : np.array(t)[:, i] for i, l in enumerate(labels)}
        })

    return Dataset.from_list(data_list)


# adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py

def create_tokenize_and_align_labels_map(tokenizer, features):
    def tokenize_and_align_labels(dataset, label_all_tokens=False):
        tokenized_inputs = tokenizer(dataset['text'], max_length=128, padding=True, truncation=True,
                                    is_split_into_words=True)
        labels = dict()
        for feature_name in features:
            labels[feature_name] = list()
            for i, label in enumerate(dataset[feature_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if label_all_tokens:
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(-100)

                    previous_word_idx = word_idx

                labels[feature_name].append(label_ids)
            tokenized_inputs[feature_name] = labels[feature_name]
        return tokenized_inputs

    return tokenize_and_align_labels