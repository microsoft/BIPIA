# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
from typing import Dict, Any


class BaseModel:
    require_system_prompt: bool

    def load_config(self, config: str | dict) -> Dict:
        if isinstance(config, dict):
            return config

        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        return config

    def process_fn(self):
        raise NotImplementedError

    def generate(self, data: Any):
        raise NotImplementedError
