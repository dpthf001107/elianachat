"""LangChain LLM wrapper for the local Mi:dm model."""

from typing import Optional

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

try:
    from app.service.midm_loader import load_midm  # EC2
except ImportError:
    from backend.app.service.midm_loader import load_midm  # Local


def build_midm_llm(model_path: Optional[str] = None, max_new_tokens: int = 256) -> HuggingFacePipeline:
    """Create a LangChain-compatible LLM from the local Mi:dm checkpoint.

    Args:
        model_path: Local model directory. If None, uses DEFAULT_MIDM_PATH from loader.
        max_new_tokens: Generation cap.

    Returns:
        HuggingFacePipeline ready to plug into LangChain chains.
    """
    tokenizer, model = load_midm(model_path) if model_path else load_midm()
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=text_gen)


