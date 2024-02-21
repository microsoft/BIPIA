# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict, Any, Tuple

import json
import jsonlines
from collections import defaultdict
import pandas as pd

from .utils import insert_end, insert_start, insert_middle

__all__ = ["BasePIABuilder", "QAPIABuilder"]


class BasePIABuilder:
    name: str
    system_prompt_template: str
    user_prompt_template: str

    def __init__(self, seed: int = None):
        self.seed = seed

    def __call__(
        self,
        contexts: str,
        attacks: str,
        insert_fns=[insert_end, insert_start, insert_middle],
        insert_fn_names=["end", "start", "middle"],
        enable_stealth: bool = False,
    ) -> pd.DataFrame:
        self.enable_stealth = enable_stealth
        self.context_samples = self.load_context(contexts)
        self.attacks = self.load_attack(attacks)

        return self.construct_samples(
            insert_fns=insert_fns, insert_fn_names=insert_fn_names
        )

    def construct_samples(self) -> pd.DataFrame:
        raise NotImplementedError

    def construct_prompt(self, example: Any) -> Tuple[str, str]:
        raise NotImplementedError

    def construct_response(self, example: Any) -> str:
        raise NotImplementedError

    def load_context(self, contexts: str | List) -> List:
        if isinstance(contexts, list):
            return contexts

        with jsonlines.open(contexts, "r") as reader:
            context_samples = list(reader.iter())

        return context_samples

    def load_attack(self, attacks: str | dict) -> Dict:
        if isinstance(attacks, dict):
            return attacks

        with open(attacks, "r") as f:
            attacks = json.load(f)

        # For the new version, each attack name have 5 prompts
        flat_attacks = {}

        for attack_name in attacks:
            for i, attack_str in enumerate(attacks[attack_name]):
                new_attack_name = f"{attack_name}-{i}"
                flat_attacks[new_attack_name] = attack_str

        # if enable stealth, the attack applies encoding to make the attack prompt harder to detect.
        # Here we inplement with base64
        if self.enable_stealth:
            import base64

            def base64_encode(m):
                encoded_bytes = base64.b64encode(m.encode("utf-8"))
                encoded_string = encoded_bytes.decode("utf-8")
                return encoded_string

            for attack_name in flat_attacks:
                orig_attack_str = flat_attacks[attack_name]
                flat_attacks[attack_name] = base64_encode(orig_attack_str)

        return flat_attacks


class QAPIABuilder(BasePIABuilder):
    def construct_samples(
        self,
        insert_fns=[insert_end, insert_start, insert_middle],
        insert_fn_names=["end", "start", "middle"],
    ) -> pd.DataFrame:
        samples = defaultdict(list)

        for insert_fn_name, insert_fn in zip(insert_fn_names, insert_fns):
            for normal_sample in self.context_samples:
                context = normal_sample["context"]
                question = normal_sample["question"]
                ideal = normal_sample["ideal"]

                for attack_name in self.attacks:
                    attack_str = self.attacks[attack_name]

                    poisoned_context = insert_fn(
                        context, attack_str, random_state=self.seed
                    )
                    samples["context"].append(poisoned_context)
                    samples["attack_name"].append(attack_name)
                    samples["attack_str"].append(attack_str)
                    samples["task_name"].append(self.name)
                    samples["ideal"].append(ideal)
                    samples["question"].append(question)
                    samples["position"].append(insert_fn_name)

        return pd.DataFrame.from_dict(samples)
