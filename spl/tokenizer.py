# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken

logger = getLogger(__name__)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    def __init__(self, encoding_name: str):
        """
        Initializes the Tokenizer with a Tiktoken encoding.

        Args:
            encoding_name (str): The name of the Tiktoken encoding.
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.pad_id = self.tokenizer.encode("<pad>")[0]  # Assign a unique ID for pad
        self.bos_id = self.tokenizer.encode("<bos>")[0]  # Assign a unique ID for BOS
        self.eos_id = self.tokenizer.encode("<eos>")[0]  # Assign a unique ID for EOS

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    def encode(self, s: str, bos: bool = False, eos: bool = False, 
               allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
               disallowed_special: Union[Literal["all"], Collection[str]] = ()) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.
        """
        tokens = self.tokenizer.encode(s, allowed_special=allowed_special, disallowed_special=disallowed_special)
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(t)


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.bos_id)
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.bos_id)
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), bos=False, eos=False))
        tokens.append(self.tokenizer.eos_id)
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.bos_id)
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens
