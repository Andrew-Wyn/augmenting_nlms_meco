from transformers import PreTrainedTokenizerBase, DataCollatorForTokenClassification
from typing import Optional, Union, List, Dict, Any
from transformers.utils import PaddingStrategy


class DataCollatorForMultiTaskTokenClassification(DataCollatorForTokenClassification):
    # def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
    #     super().__init__(tokenizer, pad_to_multiple_of)
    # tokenizer: PreTrainedTokenizerBase
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100
    # return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        return self.torch_call(features)

    def torch_call(self, features):
        import torch

        labels_names = [feat_name for feat_name in features[0].keys() if feat_name.startswith('label_')]
        no_labels_features = [{k: v for k, v in feature.items() if k not in labels_names} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        all_labels = dict()
        for label_name in labels_names:
            all_labels[label_name] = [feature[label_name] for feature in features]

        if len(all_labels) == 0:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        batch_labels_dict = dict()
        if padding_side == "right":
            for label_name in labels_names:
                labels = all_labels[label_name]
                batch_labels_dict[label_name] = [
                    to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                ]
        else:
            for label_name in labels_names:
                labels = all_labels[label_name]
                batch_labels_dict[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
                ]

        batch['labels'] = dict()
        for label_name in labels_names:
                batch['labels'][label_name[len('label_'):]] = torch.tensor(batch_labels_dict[label_name],
                                                                           dtype=torch.float32)
        return batch


class MultiLabelDataCollatorWithPadding:

    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors = 'pt'):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
    # tokenizer: PreTrainedTokenizerBase
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        labels_names = [feat_name for feat_name in features[0].keys() if feat_name.startswith('label_')]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch['labels'] = dict()
        for label_name in labels_names:
            batch['labels'][label_name[len('label_'):]] = batch[label_name]
            del batch[label_name]
   
        return batch