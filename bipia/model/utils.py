# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from transformers import StoppingCriteria

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def get_compute_capability():
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this device!")

    capability_str = torch.cuda.get_device_capability()
    capability = float(f"{capability_str[0]}.{capability_str[1]}")
    return capability

def check_bf16_support():
    capability = get_compute_capability()
    if capability >= 8.0:
        return True
    return False