"""
Shared language model loader.

Both baseline and RAG systems import from here to avoid loading the model twice.
"""

from __future__ import annotations

import torch
from config import GENERATION_MODEL_NAME, GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE

_model = None
_tokenizer = None


def get_model():
    """
    Load and cache the generation model. Returns (model, tokenizer).
    Only one copy is kept in memory regardless of how many callers use it.
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[Model] Loading: {GENERATION_MODEL_NAME} ...")
    _tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)

    if torch.cuda.is_available():
        _model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_NAME,
            torch_dtype=torch.float16,
        ).cuda()
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_NAME,
            torch_dtype=torch.float32,
        )

    print(f"[Model] Loaded on {next(_model.parameters()).device}")
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
             temperature: float = GENERATION_TEMPERATURE) -> str:
    """
    Generate text from a prompt using the shared model.
    """
    model, tokenizer = get_model()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()
