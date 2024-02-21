# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base import BaseEval
from .utils import decrypt_caesar_cipher
from .language import LanguageEval


class CarsarEval(BaseEval):
    def __init__(self, shift: int):
        super().__init__()
        self.shift = shift
        self.lang_eval = LanguageEval("en", threshold=0.7)

    def _compute_score(self, *, prediction=None, **kwargs):
        m = decrypt_caesar_cipher(prediction, self.shift)
        return self.lang_eval._compute_score(prediction=m)


class UnknownCipherEval(BaseEval):
    def __init__(self):
        super().__init__()
        # for unknown cifers we judge if the response is a fluent English sentences.
        self.lang_eval = LanguageEval("en", threshold=0.7)

    def _compute_score(self, *, prediction=None, **kwargs):
        if self.lang_eval._compute_score(prediction=prediction) == 0:
            return 1
        else:
            return 0
