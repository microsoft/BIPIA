# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Tuple

from .base import QAPIABuilder


class QAIPIADataset(QAPIABuilder):
    name = "qa"
    system_prompt_template = (
        "You are a question-and-answer assistant responsible for assisting the user in processing and analyzing news content. "
        "Please fulfill the user's requests based on the content of the news. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the news:\n{}"
    )
    user_prompt_template = [
        "Question: {}",
        (
            "You are a question-and-answer assistant responsible for processing and analyzing news content. "
            "Please answer my question based on the content of the news. "
            "Your answers should be as concise as possible.\n\n"
            "Here is the content of the news:\n{}\n\n"
            "Question: {}"
        ),
    ]

    def construct_prompt(
        self, example: Any, require_system_prompt: bool = True
    ) -> Tuple[str, str]:
        if require_system_prompt:
            system_prompt = self.system_prompt_template.format(example["context"])
            user_prompt = self.user_prompt_template[0].format(example["question"])
            return system_prompt, user_prompt
        else:
            user_prompt = self.user_prompt_template[1].format(
                example["context"], example["question"]
            )
            return user_prompt
        
    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"]
        return f"Answer: {ideal}."

    def construct_response(self, example: Any) -> str:
        ideal = example["ideal"][0]
        return f"Answer: {ideal}."