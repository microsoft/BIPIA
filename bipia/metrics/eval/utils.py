# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import re


def prompt_format_fn(prompt: str, chat: bool = False):
    if chat:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = (
            f"<|im_start|>user\n{prompt}\n<|im_end|>" + "\n<|im_start|>assistant\n"
        )
    return messages


def is_base64(s):
    try:
        if len(s) % 4 != 0:
            return False
        if base64.b64encode(base64.b64decode(s)).decode() == s:
            return True
    except Exception:
        pass
    return False


def is_base58(s):
    base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return all(c in base58_chars for c in s)


def is_base32(s):
    try:
        if len(s) % 8 != 0:
            return False
        if base64.b32encode(base64.b32decode(s)).decode() == s:
            return True
    except Exception:
        pass
    return False


def is_base16(text):
    return re.match(r"^[0-9A-Fa-f]+$", text) is not None


def decrypt_caesar_cipher(ciphertext, shift):
    plaintext = ""
    for char in ciphertext:
        if char.isalpha():
            unicode_offset = ord("a") if char.islower() else ord("A")
            decrypted_char_unicode = (
                ord(char) - unicode_offset - shift
            ) % 26 + unicode_offset
            plaintext += chr(decrypted_char_unicode)
        else:
            plaintext += char  # Keep non-alphabetic characters unchanged
    return plaintext
