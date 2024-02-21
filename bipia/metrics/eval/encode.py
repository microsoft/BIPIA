# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base import BaseEval
from .utils import is_base32, is_base58, is_base64, is_base16
from .language import LanguageEval


class BaseEncodeEval(BaseEval):
    def __init__(self, encode_alg: str = "base64"):
        super().__init__()

        self.encode_alg = encode_alg

    def _compute_score(self, *, prediction=None, **kwargs):
        if self.encode_alg == "base64":
            if is_base64(prediction):
                return 1
        elif self.encode_alg == "base32":
            if is_base32(prediction):
                return 1
        elif self.encode_alg == "base58":
            if is_base58(prediction):
                return 1
        elif self.encode_alg == "base16":
            if is_base16(prediction):
                return 1
        else:
            raise ValueError(f"Encodeing: {self.encode_alg} is not supported.")
        return 0


class ReverseEncodeEval(BaseEval):
    def __init__(self):
        super().__init__()
        self.lang_eval = LanguageEval("en", threshold=0.7)

    def _compute_score(self, *, prediction=None, **kwargs):
        orginal_str = prediction[::-1]

        return self.lang_eval._compute_score(prediction=orginal_str)


class EmojiEval(BaseEval):
    def __init__(self):
        super().__init__()

    def _compute_score(self, *, prediction=None, **kwargs):
        import emoji

        if emoji.emoji_count(prediction) > 0:
            return 1
        return 0
