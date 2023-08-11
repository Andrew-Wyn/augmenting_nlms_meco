import numpy as np
import pandas as pd
from tensorflow.keras.utils import pad_sequences
from sklearn.utils import shuffle
from anm.utils import LOGGER


class GazeDataset():
    def __init__(self, cf, tokenizer, filename):
        self.tokenizer = tokenizer
        self.filename = filename
        self.used_feature = cf.used_feature
        self.features = []

        self.text_inputs = []
        self.targets = []
        self.masks = []  # split key padding attention masks for the BERT model
        self.maps = []  # split mappings between tokens and original words

        self.feature_max = cf.feature_max if "feature_max" in cf.__dict__ else None  # gaze features will be standardized between 0 and self.feature_max

    def tokenize_and_map(self, sentence):
        """
        Tokenizes a sentence, and returns the tokens and a list of starting indices of the original words.
        """
        tokens = []
        map = []

        for w in sentence:
            map.append(len(tokens))
            tokens.extend(self.tokenizer.tokenize(w) if self.tokenizer.tokenize(w) else [self.tokenizer.unk_token])

        return tokens, map

    def tokenize_from_words(self):
        """
        Tokenizes the sentences in the dataset with the pre-trained tokenizer, storing the start index of each word.
        """
        LOGGER.info(f"Tokenizing sentences")
        tokenized = []
        maps = []

        for s in self.text_inputs:
            tokens, map = self.tokenize_and_map(s)

            tokenized.append(tokens)
            maps.append(map)
            #print(tokens)
        print("max tokenized seq len: ", max(len(l) for l in tokenized))

        self.text_inputs = tokenized
        self.maps = maps

    def calc_input_ids(self):
        """
        Converts tokens to ids for the BERT model.
        """
        LOGGER.info(f"Calculating input ids")
        ids = [self.tokenizer.prepare_for_model(self.tokenizer.convert_tokens_to_ids(s))["input_ids"]
                for s in self.text_inputs]
        self.text_inputs = pad_sequences(ids, value=self.tokenizer.pad_token_id, padding="post")

    def calc_attn_masks(self):
        """
        Calculates key paddding attention masks for the BERT model.
        """
        LOGGER.info(f"Calculating attention masks")
        self.masks = [[j != self.tokenizer.pad_token_id for j in i] for i in self.text_inputs]

    def read_pipeline(self):
        # retrieve the sentences and the targets from the dataset.
        self.load_data()

        # retrieve the output dimension
        self.d_out = len(self.targets[0][0])  # number of gaze features
        self.target_pad = -1

        # Tokenize the input and retrieving the masking
        self.tokenize_from_words()
        # Pad the targets
        self.pad_targets()
        # Prepare inputs for model
        self.calc_input_ids()
        # Compute the masks
        self.calc_attn_masks()
        # Convert the data to numpy arrays
        self.calc_numpy()

    def _create_senteces_from_data(self, data):

        dropping_cols = {"sentnum", "ia", "lang", "trialid", "ianum", "uniform_id"}
        
        # sort by sentnum and ianum, to avoid splitted sentences
        data = data.sort_values(by=["sentnum", "ianum"])

        # create sentence_id
        data["sentence_id"] = data["trialid"].astype(int).astype(str) + data["sentnum"].astype(int).astype(str)

        self.features = [e for e in list(data.columns) if e not in dropping_cols]

        word_func = lambda s: [w for w in s["ia"].values.tolist()]

        if not self.used_feature is None and self.used_feature in self.features:
            features_func = lambda s: [np.array([s.drop(columns=dropping_cols).iloc[i, self.features.index(self.used_feature)]])
                                    for i in range(len(s))]
        else:
            features_func = lambda s: [np.array(s.drop(columns=dropping_cols).iloc[i])
                                    for i in range(len(s))]

        sentences = data.groupby("sentence_id").apply(word_func).tolist()

        targets = data.groupby("sentence_id").apply(features_func).tolist()

        return sentences, targets

    def load_data(self):
        LOGGER.info(f"Loading data")
        
        dataset = pd.read_csv(self.filename, index_col=0)

        sentences, targets = self._create_senteces_from_data(dataset)

        self.text_inputs = sentences
        self.targets = targets

        LOGGER.info(f"Lenght of data : {len(self.text_inputs)}")

    def pad_targets(self):
        """
        Adds the pad tokens in the positions of the [CLS] and [SEP] tokens, adds the pad
        tokens in the positions of the subtokens, and pads the targets with the pad token.
        """
        LOGGER.info(f"Padding targets")
        targets = [np.full((len(i), self.d_out), self.target_pad, dtype=np.float16) for i in self.text_inputs]
        for k, (i, j) in enumerate(zip(self.targets, self.maps)):
            targets[k][j, :] = i

        target_pad_vector = np.full((1, self.d_out), self.target_pad)
        targets = [np.concatenate((target_pad_vector, i, target_pad_vector)) for i in targets]

        self.targets = pad_sequences(targets, value=self.target_pad, padding="post", dtype="float16")

    def calc_numpy(self):
        LOGGER.info(f"Calculating numpy arrays")
        self.text_inputs = np.asarray(self.text_inputs, dtype=np.int64)
        self.masks = np.asarray(self.masks, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def randomize_data(self):
        LOGGER.info(f"Randomize numpy arrays")
        shuffled_ids = shuffle(range(self.text_inputs.shape[0]), random_state=42)
        self.text_inputs = self.text_inputs[shuffled_ids]
        self.targets = self.targets[shuffled_ids]
        self.masks = self.masks[shuffled_ids]
