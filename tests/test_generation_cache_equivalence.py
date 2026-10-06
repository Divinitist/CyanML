from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ANSWER_PATH = (
    Path(__file__).resolve().parents[1]
    / "questions"
    / "37_autoregressive_generation"
    / "answer.py"
)


def load_answer():
    spec = importlib.util.spec_from_file_location("generation_answer", ANSWER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.autoregressive_generate


def test_cache_and_full_recomputation_produce_the_same_generation() -> None:
    generate = load_answer()
    rng = np.random.default_rng(370037)
    for index in range(20):
        heads = 1 + index % 3
        head_dim = 2 + index % 3
        model_dim = heads * head_dim
        vocab_size = max(7, model_dim + 4)
        embedding = rng.normal(scale=0.3, size=(vocab_size, model_dim))
        params = {
            key: rng.normal(scale=0.2, size=(model_dim, model_dim))
            for key in ("w_q", "w_k", "w_v", "w_o")
        }
        prompt = rng.integers(0, vocab_size, size=1 + index % 5)
        arguments = (prompt, embedding, params, heads, 5, vocab_size - 1)

        full_tokens, full_logits = generate(*arguments, False)
        cached_tokens, cached_logits = generate(*arguments, True)

        assert cached_tokens == full_tokens
        np.testing.assert_allclose(cached_logits, full_logits, rtol=1e-8, atol=1e-10)
