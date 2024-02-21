# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict
import random
from collections import defaultdict

from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import PreTrainedTokenizer, BatchEncoding

def no_insert(context: str, attack: str, random_state: int = None):
    return context

def insert_end(context: str, attack: str, random_state: int = None):
    return "\n".join([context, attack])


def insert_start(context: str, attack: str, random_state: int = None):
    return "\n".join([attack, context])


def insert_middle(context: str, attack: str, random_state: int = None):
    rng = random.Random(random_state)
    sentence_indexes = list(PunktSentenceTokenizer().span_tokenize(context))
    start, _ = rng.sample(sentence_indexes, k=1)[0]

    return "\n".join([context[:start], attack, context[start:]])

def remove_none_name(messages):
    if isinstance(messages, list):
        for message in messages:
            if "name" in message and message["name"] is None:
                del message["name"]
    return messages


class DefaultDataCollator:
    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)

        for example in batch_examples:
            for key in example:
                if key == "message":
                    example[key] = remove_none_name(example[key])
                batch_rslt[key].append(example[key])

        return batch_rslt


class DataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding="longest",
        max_length=None,
        pad_to_multiple_of=None,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.return_tensors = return_tensors

    def __call__(self, batch_examples: List) -> Dict:
        batch_rslt = defaultdict(list)
        for example in batch_examples:
            for key in example:
                if key not in self.tokenizer.model_input_names:
                    batch_rslt[key].append(example[key])

        features = []
        for example in batch_examples:
            features.append(
                BatchEncoding({k: example[k] for k in self.tokenizer.model_input_names})
            )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors=self.return_tensors,
            verbose=True,
        )

        batch_rslt.update(features)
        return batch_rslt
