"""Unit tests for cutter_pipeline.outline_openai.

We don't hit the real OpenAI API — we stub the client so we can verify the
function decodes the response correctly and surfaces missing-key errors.
"""
from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutter_pipeline import outline_openai


def test_missing_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        outline_openai.generate_outline_png("a heart", str(tmp_path / "out.png"))


def test_generate_outline_png_decodes_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    payload = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"
    b64 = base64.b64encode(payload).decode()

    captured: dict = {}

    class _FakeImages:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json=b64)])

    class _FakeClient:
        def __init__(self, *_, **__):
            self.images = _FakeImages()

    monkeypatch.setattr(outline_openai, "OpenAI", _FakeClient)

    out = tmp_path / "out.png"
    result = outline_openai.generate_outline_png("a smiling sun", str(out))

    assert Path(result) == out
    assert out.read_bytes() == payload
    # Verify the prompt the API saw includes our subject and the instructional preamble.
    assert "smiling sun" in captured["prompt"]
    assert captured["size"] == "1024x1024"
    assert captured["model"] == outline_openai.IMAGE_MODEL


def test_generate_outline_creates_missing_parent_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = b64 = base64.b64encode(b"x").decode()

    class _FakeImages:
        def generate(self, **_):
            return SimpleNamespace(data=[SimpleNamespace(b64_json=b64)])

    class _FakeClient:
        def __init__(self, *_, **__):
            self.images = _FakeImages()

    monkeypatch.setattr(outline_openai, "OpenAI", _FakeClient)

    nested = tmp_path / "a" / "b" / "out.png"
    outline_openai.generate_outline_png("x", str(nested))
    assert nested.is_file()
