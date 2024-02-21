# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict

from .code import CodeIPIABuilder
from .email import EmailIPIABuilder
from .qa import QAIPIADataset
from .table import TableIPIABuilder
from .abstract import AbstractIPIADataset

from .utils import DefaultDataCollator, DataCollatorWithPadding


BUILDER_TO_CLASS = OrderedDict(
    [
        ("code", CodeIPIABuilder),
        ("qa", QAIPIADataset),
        ("table", TableIPIABuilder),
        ("email", EmailIPIABuilder),
        ("abstract", AbstractIPIADataset),
    ]
)


class AutoPIABuilder:
    @classmethod
    def from_name(cls, name: str):
        builder_cls = BUILDER_TO_CLASS[name]
        return builder_cls
