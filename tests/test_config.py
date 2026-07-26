"""Config resolver tests (P0-2): priority, validation, env loading."""
import pytest

import adapter.adapter_config as m


def test_defaults_resolve_to_vllm():
    cfg = m.resolve({"platform": "linux-rocm"})
    assert cfg.backend == "vllm"
    assert cfg.temperature == 0.0
    assert cfg.batch_size == 8
    assert cfg.num_shards == 1 and cfg.shard_index == 0


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")
    monkeypatch.setenv("OVISOCR2_BATCH_SIZE", "4")
    cfg = m.resolve({"platform": "linux-rocm"})
    assert cfg.backend == "smoke" and cfg.batch_size == 4


def test_cli_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")
    cfg = m.resolve({"platform": "linux-rocm", "backend": "vllm"})
    assert cfg.backend == "vllm"


def test_cli_none_does_not_clobber_env(monkeypatch):
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")
    cfg = m.resolve({"platform": "linux-rocm", "backend": None})
    assert cfg.backend == "smoke"


def test_env_local_loaded_shell_wins(tmp_path, monkeypatch):
    env_local = tmp_path / ".env.local"
    env_local.write_text('OVISOCR2_BACKEND=vllm\nOVISOCR2_MAX_TOKENS=999\n')
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")  # shell already set -> wins
    m.load_env_local(env_local)
    cfg = m.resolve({"platform": "linux-rocm"})
    assert cfg.backend == "smoke"            # shell wins over .env.local
    assert cfg.max_tokens == 999             # .env.local fills an unset key


def test_invalid_backend_rejected():
    with pytest.raises(SystemExit):
        m.resolve({"platform": "linux-rocm", "backend": "bogus"})


def test_windows_hip_plus_vllm_rejected():
    with pytest.raises(SystemExit):
        m.resolve({"platform": "windows-hip", "backend": "vllm"})


def test_bad_shard_args_rejected():
    with pytest.raises(SystemExit):
        m.resolve({"platform": "linux-rocm", "num_shards": 2, "shard_index": 2})
    with pytest.raises(SystemExit):
        m.resolve({"platform": "linux-rocm", "num_shards": 0})
