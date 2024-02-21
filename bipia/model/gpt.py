# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Any, Callable, Tuple

import os
import re
import time
import openai
from openai.error import (
    RateLimitError,
    InvalidRequestError,
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    APIError,
)

from accelerate.logging import get_logger

from .base import BaseModel

__all__ = ["GPTModel", "GPT35", "GPT4"]

logger = get_logger(__name__)


def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1


class GPTModel(BaseModel):
    def __init__(self, *, config: str | dict = None, **kwargs):
        config = self.load_config(config)
        self.config = config

    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        success = False
        while not success:
            try:
                response = openai.ChatCompletion.create(
                    api_key=self.config.get("api_key", None),
                    api_base=self.config.get("api_base", None),
                    api_type=self.config.get("api_type", None),
                    api_version=self.config.get("api_version", None),
                    engine=self.config.get("engine", None),
                    model=self.config.get("model", None),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                success = True
            except RateLimitError as e:
                # logger.warning(e, exc_info=True)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except Timeout as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except ServiceUnavailableError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except InvalidRequestError as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"choices": []}
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"choices": []}
        try:
            rslts = [i["message"]["content"] for i in response["choices"]]
        except Exception as e:
            logger.warning(e, exc_info=True)
            rslts = []

        return rslts

    def completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        success = False
        while not success:
            try:
                response = openai.Completion.create(
                    api_key=self.config.get("api_key", None),
                    api_base=self.config.get("api_base", None),
                    api_type=self.config.get("api_type", None),
                    api_version=self.config.get("api_version", None),
                    engine=self.config.get("engine", None),
                    model=self.config.get("model", None),
                    prompt=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                success = True
            except RateLimitError as e:
                # logger.warning(e, exc_info=True)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except Timeout as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except ServiceUnavailableError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except InvalidRequestError as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"choices": []}
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = {"choices": []}

        rslts = [i["text"] for i in response["choices"]]
        return rslts

    def generate(self, data: Any, **kwargs):
        temperature = kwargs.pop("temperature", 0)
        if self.config["chat"]:
            rslts = []
            for message in data["message"]:
                rslt = self.chat_completion(message, temperature=temperature)
                rslts.extend(rslt)
        else:
            rslts = self.completion(data["message"], temperature=temperature)
        return rslts


class GPTModelWSystem(GPTModel):
    require_system_prompt = True

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

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35(GPTModelWSystem):
    pass


class GPT4(GPTModelWSystem):
    pass


class GPTModelWOSystem(GPTModel):
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
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35WOSystem(GPTModelWOSystem):
    pass


class GPT4WOSystem(GPTModelWOSystem):
    pass
