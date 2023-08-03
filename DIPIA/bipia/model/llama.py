from typing import Any, Callable, Tuple, Dict, List

import transformers
from transformers import AutoTokenizer, GenerationConfig, StoppingCriteriaList

from .llm import LLMModel
from .utils import EndOfFunctionCriteria

__all__ = [
    "LLAMAModel",
    "Alpaca",
    "Vicuna",
    "Baize",
    "StableVicuna",
    "Koala",
]


class LLAMAModel(LLMModel):
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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class Alpaca(LLAMAModel):
    require_system_prompt = False

    def apply_delta(self):
        # aplaca released model changes pad token and its embedding
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["delta_weights"], use_fast=False
        )

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.tokenizer.padding_side = "left"
        return super().apply_delta()

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
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{user_prompt}\n\n"
            "### Response:\n"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example


class Vicuna(LLAMAModel):
    """The 1.1 version of Vicuna"""

    require_system_prompt = False

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
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {user_prompt} "
            f"ASSISTANT:"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example


class Baize(LLAMAModel):
    require_system_prompt = False

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
        example["target"] = example["ideal"]
        user_prompt = prompt_construct_fn(example)

        message = (
            "The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). "
            "Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. "
            "The human and the AI assistant take turns chatting. "
            "Human statements start with [|Human|] and AI assistant statements start with [|AI|]. "
            "The AI assistant always provides responses in as much detail as possible, and in Markdown format. "
            "The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. "
            "Complete the transcript in exactly that format.\n"
            "[|Human|]Hello!\n"
            "[|AI|]Hi!\n"
            f"[|Human|]{user_prompt}\n"
            "[|AI|]"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_stopping_criteria(self, input_ids):
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(input_ids.shape[1], ["[|Human|]"], self.tokenizer)]
        )
        return stopping_criteria

    def post_process(self, responses: List[str]):
        truncated_responses = []

        for response in responses:
            index = response.find("[|Human|]")

            if index != -1:
                response = response[:index]
            else:
                response = response
            response = response.strip()
            truncated_responses.append(response)

        return truncated_responses


class StableVicuna(Alpaca):
    require_system_prompt = False

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
        example["target"] = example["ideal"]
        user_prompt = prompt_construct_fn(example)
        message = (
            "### Assistant: I am StableVicuna, a large language model created by CarperAI. "
            "I am here to chat!\n\n"
            f"### Human: {user_prompt}\n"
            "### Assistant: "
        )
        example["message"] = message
        example.update(self.tokenizer(message))

        return example

    def load_stopping_criteria(self, input_ids):
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(input_ids.shape[1], ["###"], self.tokenizer)]
        )
        return stopping_criteria

    def post_process(self, responses: List[str]):
        truncated_responses = []

        for response in responses:
            index = response.find("###")

            if index != -1:
                response = response[:index]
            else:
                response = response
            response = response.strip()
            truncated_responses.append(response)

        return truncated_responses


class Koala(LLAMAModel):
    require_system_prompt = False

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
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = f"BEGINNING OF CONVERSATION: USER: {user_prompt} GPT:"

        example["message"] = message
        example.update(self.tokenizer(message))

        return example


class Wizard(Vicuna):
    def apply_delta(self):
        # aplaca released model changes pad token and its embedding
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["delta_weights"], use_fast=False
        )

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.tokenizer.padding_side = "left"
        return super().apply_delta()


class GPT4ALL(LLAMAModel):
    require_system_prompt = False

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
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = (
            "### Instruction:\n"
            "The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.\n"
            f"### Prompt: {user_prompt}\n"
            "### Response:"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example

    def load_stopping_criteria(self, input_ids):
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(input_ids.shape[1], ["###"], self.tokenizer)]
        )
        return stopping_criteria

    def post_process(self, responses: List[str]):
        truncated_responses = []

        for response in responses:
            index = response.find("###")

            if index != -1:
                response = response[:index]
            else:
                response = response
            response = response.strip()
            truncated_responses.append(response)

        return truncated_responses


class Guanaco(StableVicuna):
    require_system_prompt = False

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
        user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        message = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            f"### Human: {user_prompt}\n"
            f"### Assistant:"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example