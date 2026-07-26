"""Root conftest — makes the repo importable + test infrastructure.

* A root-level conftest puts the repo root on sys.path so tests can
  ``import adapter.<module>``.
* ``_isolate_env`` snapshots/restores os.environ around every test so the
  adapter's ``load_env_local`` (which writes os.environ) cannot leak across
  tests.
* ``fake_vllm`` is a FACTORY fixture: ``fake_vllm(**llm_kwargs)`` installs a
  configured ``FakeLLM`` as ``vllm.LLM`` so the adapter's vllm branch runs on
  CPU. The smoke path must never import vllm and does not use it.
"""
from __future__ import annotations

import os
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot/restore os.environ so OVISOCR2_* mutations stay test-local."""
    snapshot = os.environ.copy()
    yield
    for k in [k for k in os.environ if k not in snapshot]:
        del os.environ[k]
    for k, v in snapshot.items():
        os.environ[k] = v


class _FakeOutput:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeOutput(text)]


class FakeLLM:
    """Stub ``vllm.LLM``. Absorbs any constructor kwargs (gdn_prefill_backend, model, ...).

    texts: ordered per-page outcomes (str | Exception), consumed in order across
           every generate() input; when empty/exhausted, ``default`` is returned.
    batch_fail: generate() raises immediately (simulates a whole-batch OOM).
    """

    def __init__(self, *args, texts=None, default="# body\n\ntext", batch_fail=False, **kwargs):
        self._texts = list(texts) if texts else []
        self._default = default
        self._batch_fail = batch_fail

    def generate(self, inputs, sp):
        if self._batch_fail:
            raise RuntimeError("simulated batch OOM")
        out = []
        for _ in inputs:
            if self._texts:
                t = self._texts.pop(0)
                if isinstance(t, Exception):
                    raise t
                out.append(_FakeRequestOutput(t))
            else:
                out.append(_FakeRequestOutput(self._default))
        return out

    def get_tokenizer(self):
        class _T:
            def apply_chat_template(self, msgs, **kw):
                return "<|prompt|>"
        return _T()


class _FakeSamplingParams:
    def __init__(self, *a, **k):
        pass


@pytest.fixture
def fake_vllm(monkeypatch):
    """Factory: ``fake_vllm(**llm_kwargs)`` installs a configured FakeLLM as vllm.LLM."""

    def install(**llm_kwargs):
        def _make_llm(*a, **k):
            return FakeLLM(**llm_kwargs)

        mod = types.ModuleType("vllm")
        mod.LLM = _make_llm
        mod.SamplingParams = _FakeSamplingParams
        monkeypatch.setitem(sys.modules, "vllm", mod)
        # reset adapter memoization so each test builds a fresh FakeLLM
        import adapter.run_adapter as R

        R._LLM = None
        R._CHAT = None
        return mod

    return install
