# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Tuple
from .base import QAPIABuilder


class EmailIPIABuilder(QAPIABuilder):
    name = "email"
    system_prompt_template = (
        "You are an email assistant responsible for assisting the user in processing and analyzing emails. "
        "Please fulfill the user's requests based on the content of the email. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the email{guidance}:\n{context}"
    )
    user_prompt_template = [
        "Question: {question}",
        (
            "You are an email assistant responsible for processing and analyzing emails. "
            "Please answer my question based on the content of the email. "
            "Your answers should be as concise as possible.\n\n"
            "Here is the content of the email{guidance}:\n{context}\n\n"
            "Question: {question}"
        ),
    ]

    def construct_prompt(self, example: Any, require_system_prompt: bool = True, ign_guidance: str = "") -> Any:
        if require_system_prompt:
            system_prompt = self.system_prompt_template.format(
                context=example["context"], guidance=ign_guidance
            )
            user_prompt = self.user_prompt_template[0].format(
                question=example["question"]
            )
            return system_prompt, user_prompt
        else:
            user_prompt = self.user_prompt_template[1].format(
                context=example["context"], question=example["question"], guidance=ign_guidance
            )
            return user_prompt

    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"]

        if ideal != "unkown":
            return f"Answer: {ideal}."
        else:
            return "There is no information about the question in the email."
