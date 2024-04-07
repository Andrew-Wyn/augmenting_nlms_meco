import numpy as np
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
from anm.gaze_dataloader.datacollator import DataCollatorForMultiTaskTokenClassification
from torch.utils.data import DataLoader

def _create_senteces_from_data(data, tasks, keep_id=False):
    dropping_cols = set(data.columns).difference(set(tasks))
    
    # sort by trialid, sentnum and ianum, to avoid splitted sentences
    data = data.sort_values(by=["trialid", "sentnum", "ianum"])

    # create sentence_id
    data["sentence_id"] = data["trialid"].astype(int).astype(str) + '_' +data["sentnum"].astype(int).astype(str)
    
    dropping_cols.add("sentence_id")
    
    word_func = lambda s: [str(w) for w in s["ia"].values.tolist()]

    features_func = lambda s: [np.array(s.drop(columns=dropping_cols).iloc[i])
                            for i in range(len(s))]

    grouped_data = data.groupby("sentence_id")

    sentences_ids = list(grouped_data.groups.keys())
    sentences = grouped_data.apply(word_func).tolist()
    targets = grouped_data.apply(features_func).tolist()

    data_list = []
    
    for s_id, s, t in zip(sentences_ids, sentences, targets):
        if keep_id:
            data_list.append({
            **{"id": s_id},
            **{"text": s,},
            **{"label_"+str(l) : np.array(t)[:, i] for i, l in enumerate(tasks)}
        })
        else:
            data_list.append({
                **{"text": s,},
                **{"label_"+str(l) : np.array(t)[:, i] for i, l in enumerate(tasks)}
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


def _create_and_fit_sclers(dataset, features):
    # create and fit the scalers

    scalers = {}

    for feat in features:
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(np.array([el for sen in dataset[feat] for el in sen ]).reshape(-1, 1))
        scalers[feat] = scaler

    return scalers


def minmax_preprocessing(cf, dataset, tokenizer, train_test_split:tuple = None):
    features = [col_name for col_name in dataset.column_names if col_name.startswith('label_')]

    if not train_test_split is None:
        train_idx, test_idx = train_test_split
        train_ds = dataset.select(train_idx)
        test_ds = dataset.select(test_idx)

        # create and fit the scalers
        scalers = _create_and_fit_sclers(train_ds, features)
        # preprocessing function
        def minmaxscaling_function(row):
            for k, scaler in scalers.items():
                row[k] = scaler.transform(np.array(row[k]).reshape(-1, 1)).squeeze().tolist()

            return row

        train_ds = train_ds.map(minmaxscaling_function)
        test_ds = test_ds.map(minmaxscaling_function)

        tokenized_dataset_train = train_ds.map(
                    create_tokenize_and_align_labels_map(tokenizer, features),
                    batched=True,
                    remove_columns=train_ds.column_names,
                    # desc="Running tokenizer on dataset",
        )
        tokenized_dataset_test = test_ds.map(
                    create_tokenize_and_align_labels_map(tokenizer, features),
                    batched=True,
                    remove_columns=test_ds.column_names,
                    # desc="Running tokenizer on dataset",
        )

        data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
        train_dl = DataLoader(tokenized_dataset_train, shuffle=True, collate_fn=data_collator, batch_size=cf.train_bs)
        test_dl = DataLoader(tokenized_dataset_test, shuffle=True, collate_fn=data_collator, batch_size=cf.test_bs)
        return train_dl, test_dl
    else:
        # create and fit the scalers
        scalers = _create_and_fit_sclers(dataset, features)
        # preprocessing function
        def minmaxscaling_function(row):
            for k, scaler in scalers.items():
                row[k] = scaler.transform(np.array(row[k]).reshape(-1, 1)).squeeze().tolist()

            return row

        dataset = dataset.map(minmaxscaling_function)

        tokenized_dataset = dataset.map(
                    create_tokenize_and_align_labels_map(tokenizer, features),
                    batched=True,
                    remove_columns=dataset.column_names,
                    # desc="Running tokenizer on dataset",
        )

        data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
        dl = DataLoader(tokenized_dataset, shuffle=True, collate_fn=data_collator, batch_size=cf.train_bs)
        return dl