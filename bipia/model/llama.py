# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List

from transformers import AutoTokenizer

from vllm import LLM
from .utils import check_bf16_support
from .vllm_worker import vLLMModel

__all__ = [
    "LLAMAModel",
    "Alpaca",
    "Vicuna",
    "Baize",
    "StableVicuna",
    "Koala",
    "Wizard",
    "GPT4ALL",
    "Guanaco",
    "Llama2",
]


class LLAMAModel(vLLMModel):
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=False
        )

        if self.tokenizer.pad_token is None:
            # LLAMA doesnot have pad token (https://github.com/huggingface/transformers/issues/22312)
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer

    def load_model(self):
        tensor_parallel_size = self.kwargs.get("tensor_parallel_size", 1)
        if check_bf16_support():
            dtype = "bfloat16"
        else:
            dtype = "float16"
        self.model = LLM(
            model=self.config["model_name"],
            trust_remote_code=self.config.get("trust_remote_code", False),
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            tokenizer_mode="slow",
        )
        return self.model


class Alpaca(LLAMAModel):
    require_system_prompt = False


class Vicuna(LLAMAModel):
    require_system_prompt = False


class Baize(LLAMAModel):
    require_system_prompt = False


class StableVicuna(LLAMAModel):
    require_system_prompt = False


class Koala(LLAMAModel):
    require_system_prompt = False


class Wizard(LLAMAModel):
    require_system_prompt = False


class GPT4ALL(LLAMAModel):
    require_system_prompt = False


class Guanaco(LLAMAModel):
    require_system_prompt = False


class Llama2(LLAMAModel):
    require_system_prompt = False
