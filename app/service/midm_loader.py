"""Utilities to load the Mi:dm LLM locally."""

from functools import lru_cache
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MIDM_PATH = "backend/app/model/midm"


@lru_cache(maxsize=1)
def load_midm(model_path: str = DEFAULT_MIDM_PATH) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the Mi:dm model and tokenizer from a local directory.

    Args:
        model_path: Local path to the Mi:dm checkpoint directory containing
            `model.safetensors`, `tokenizer.json`, `config.json`, etc.

    Returns:
        (tokenizer, model) tuple loaded with trust_remote_code enabled.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    return tokenizer, model


