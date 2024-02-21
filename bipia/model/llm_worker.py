# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Callable, Tuple, List
import sys
import gc
import os

import torch

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteriaList,
    AutoModelForSeq2SeqLM,
)
from peft import PeftModel

import fastchat.model

from .base import BaseModel
from .utils import EndOfFunctionCriteria


__all__ = ["LLMModel", "RwkvModel", "ChatGLM", "OASST", "FastChatT5"]


class LLMModel(BaseModel):
    def __init__(self, *, config: str | dict = None, **kwargs):
        self.config = self.load_config(config)

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

        self.generation_config = self.load_generation_config()

    def apply_lora(self):
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config["lora_weights"],
            torch_dtype=torch.float16,
        )

    @torch.no_grad()
    def apply_delta(self):
        # load delta to cpu memory to avoid unecessary cuda memory usage
        delta = AutoModelForCausalLM.from_pretrained(
            self.config["delta_weights"],
            load_in_8bit=self.config["load_8bit"],
            torch_dtype=torch.float16,
            device_map={"": torch.device("cpu")},
            low_cpu_mem_usage=True,
        )

        for name, param in self.model.state_dict().items():
            assert name in delta.state_dict(), f"Weight {name} not in model parameters."
            param.data += delta.state_dict()[name].to(param.data.device)

        # need gc.collect() (issue https://github.com/huggingface/transformers/issues/22801)
        del delta
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if "lora_weights" in self.config:
            self.apply_lora()

        if "delta_weights" in self.config:
            self.apply_delta()

        if not self.config["load_8bit"]:
            self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def get_conv_template(self):
        conv_template = fastchat.model.get_conversation_template(
            self.config.get("template_name", self.config["model_name"])
        )
        return conv_template

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=False,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )
        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_generation_config(self):
        # Temperature = 0 will get error, direct disable do sample
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_stopping_criteria(self, input_ids):
        conv_template = self.get_conv_template()

        if conv_template.stop_str is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    EndOfFunctionCriteria(
                        input_ids.shape[1], [conv_template.stop_str], self.tokenizer
                    )
                ]
            )
        else:
            stopping_criteria = None
        return stopping_criteria

    @torch.no_grad()
    def generate(self, data):
        input_ids = torch.as_tensor(data["input_ids"]).cuda()

        stopping_criteria = self.load_stopping_criteria(input_ids)

        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria,
        )

        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)
        return responses

    def post_process(self, responses: List[str]):
        conv_template = self.get_conv_template()
        if conv_template.stop_str is not None:
            truncated_responses = []
            for response in responses:
                index = response.find(conv_template.stop_str)

                if index != -1:
                    response = response[:index]
                else:
                    response = response
                response = response.strip()
                truncated_responses.append(response)

            return truncated_responses
        else:
            return [i.strip() for i in responses]

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        conv_template = self.get_conv_template()
        if self.require_system_prompt:
            system_prompt, user_prompt = prompt_construct_fn(example)
            conv_template.set_system_message(system_prompt)
        else:
            user_prompt = prompt_construct_fn(example)

        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        example["message"] = conv_template.get_prompt()

        example.update(self.tokenizer(example["message"]))
        return example


class RwkvModel(LLMModel):
    require_system_prompt = False

    def load_model(self):
        import fastchat.model.rwkv_model
        from huggingface_hub import hf_hub_download
        import tempfile

        rwkv_model_file = "RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192.pth"
        temp_dir = tempfile.TemporaryDirectory().name
        hf_hub_download(
            self.config["model_name"],
            filename=rwkv_model_file,
            local_dir=temp_dir,
        )
        self.model_path = os.path.join(temp_dir, rwkv_model_file)
        self.model = fastchat.model.rwkv_model.RwkvModel(self.model_path)
        return self.model

    def load_tokenizer(self):
        revision = "main"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-160m", revision=revision
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"
            self.tokenizer.pad_token_id = 1

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer

    def generate(self, data):
        from transformers import AutoTokenizer

        from fastchat.serve.inference import generate_stream
        from fastchat.conversation import get_conv_template
        from fastchat.utils import get_context_length

        context_len = get_context_length(self.model.config)

        responses = []
        for prompt in data["message"]:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "EleutherAI/pythia-160m", use_fast=True
                )

            conv = get_conv_template("rwkv")

            gen_params = {
                "model": self.model_path,
                "prompt": prompt,
                "temperature": self.generation_config.temperature,
                "repetition_penalty": self.generation_config.repetition_penalty,
                "max_new_tokens": self.generation_config.max_new_tokens,
                "stop": conv.stop_str,
                "stop_token_ids": conv.stop_token_ids,
                "echo": False,
            }
            res_iter = generate_stream(
                self.model, self.tokenizer, gen_params, "cuda", context_len
            )

            for res in res_iter:
                pass

            output = res["text"]
            responses.append(output)

        return responses


class OASST(LLMModel):
    require_system_prompt = False

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"
            self.tokenizer.pad_token_id = 1

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer


class ChatGLM(LLMModel):
    require_system_prompt = False

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if not self.config["load_8bit"]:
            self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], trust_remote_code=True
        )
        self.tokenizer.model_input_names = ["input_ids"]
        self.tokenizer.padding_side = "left"
        return self.tokenizer


class FastChatT5(LLMModel):
    require_system_prompt = False

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_tokens=2048,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        return self.model

    @torch.no_grad()
    def generate(self, data):
        input_ids = torch.as_tensor(data["input_ids"]).cuda()

        stopping_criteria = self.load_stopping_criteria(input_ids)

        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria,
        )

        responses = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        responses = self.post_process(responses)
        return responses
