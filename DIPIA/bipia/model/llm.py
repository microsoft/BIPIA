from typing import Any, Callable, Tuple, List
import sys
import gc

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteriaList,
)
from peft import PeftModel

from .base import BaseModel
from .utils import EndOfFunctionCriteria

__all__ = [
    "LLMModel",
    "Dolly",
    "OASST",
    "ChatGLM",
    "StableLM",
    "MPT",
    "FastChatT5",
    "RWKV",
]


class LLMModel(BaseModel):
    def __init__(self, *, config: str|dict = None, **kwargs):
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

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=False
        )
        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_stopping_criteria(self, input_ids):
        return None

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
        return [i.strip() for i in responses]


class Dolly(LLMModel):
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
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{user_prompt}\n\n"
            "### Response:\n"
        )

        example["message"] = message

        example.update(self.tokenizer(message))
        return example


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

        message = f"<|prompter|>{user_prompt}<|endoftext|><|assistant|>"

        example["message"] = message

        example.update(self.tokenizer(message))
        return example


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

        message = f"[Round 0]\n问：{user_prompt}\n答："
        example["message"] = message

        example.update(self.tokenizer(message))

        return example


class StableLM(LLMModel):
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

        system_prompt = (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n"
            "- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.\n"
            "- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n"
            "- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.\n"
            "- StableLM will refuse to participate in anything that could harm a human.\n"
        )

        user_prompt = prompt_construct_fn(example)
        message = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
        example["message"] = message

        example.update(self.tokenizer(message))

        return example

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[50278, 50279, 50277, 1, 0] + [self.tokenizer.eos_token_id],
        )
        return self.generation_config


class MPT(LLMModel):
    require_system_prompt = True

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
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

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"], use_fast=True, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"
            self.tokenizer.pad_token_id = 1

        self.tokenizer.padding_side = "left"  # Allow batched inference
        return self.tokenizer

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[50278, 0] + [self.tokenizer.eos_token_id],
        )
        return self.generation_config

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
        system_prompt, user_prompt = prompt_construct_fn(example)
        example["target"] = example["ideal"]

        system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
        user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format("user", user_prompt)

        message = system_message + user_message + "\n<|im_start|>assistant\n"
        example["message"] = message

        example.update(self.tokenizer(message))
        return example


class FastChatT5(LLMModel):
    require_system_prompt = False

    def load_generation_config(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
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
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            f"### Human: Got any creative ideas for a 10 year old’s birthday?\n"
            f"### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
            "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
            "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n"
            "3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n"
            "4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n"
            "5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n"
            "6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n"
            "7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n"
            "8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\n"
            "Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n"
            f"### Human: {user_prompt}\n"
            f"### Assistant:"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example

    def load_stopping_criteria(self, input_ids):
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(input_ids.shape[1], ["###"], self.tokenizer)]
        )
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

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
        responses = self.post_process(responses)
        return responses

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


class RwkvModel(LLMModel):
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

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            context_length=2048,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        return self.model

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
        user_prompt = prompt_construct_fn(example).replace("\r\n", "\n").replace("\n\n", "\n")
        example["target"] = example["ideal"]

        message = (
            "Bob: hi\n\n"
            "Alice: Hi. I am your assistant and I will provide expert full response in full details. "
            "Please feel free to ask any question and I will always answer it.\n\n"
            f"Bob: {user_prompt}\n\n"
            "Alice:"
        )

        example["message"] = message
        example.update(self.tokenizer(message))

        return example

    def load_stopping_criteria(self, input_ids):
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(input_ids.shape[1], ["\n\n"], self.tokenizer)]
        )
        return stopping_criteria

    def post_process(self, responses: List[str]):
        truncated_responses = []

        for response in responses:
            index = response.find("\n\n")

            if index != -1:
                response = response[:index]
            else:
                response = response
            response = response.strip()
            truncated_responses.append(response)

        return truncated_responses