"""空白的模型环境变量不算已经选了模型，应回落到内置默认值。"""
from __future__ import annotations

import pytest

from config import LLMSettings


@pytest.mark.parametrize(
    ("env_name", "field_name", "default_model"),
    [
        ("QWEN_TEXT_MODEL", "text_model", "qwen3.7-plus"),
        ("QWEN_VISION_MODEL", "vision_model", "qwen3.8-omni-flash"),
    ],
)
def test_blank_model_env_uses_builtin_default(monkeypatch, env_name, field_name, default_model):
    monkeypatch.setenv(env_name, "")
    llm = LLMSettings(_env_file=None)

    assert getattr(llm, field_name) == default_model
