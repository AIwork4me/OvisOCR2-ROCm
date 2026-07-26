# OvisOCR2-ROCm P0 Engineering Backbone — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden OvisOCR2-ROCm's P0 engineering backbone — working install path, config that never silently falls back to `smoke`, batched/resumable inference, deterministic multi-GPU sharding, a CI gate that runs tests, and honest forward-looking telemetry — without touching committed results or the 95.88 score.

**Architecture:** Pure, fully-tested helpers (`adapter/adapter_config.py` resolver; `adapter/postprocess.py`; `adapter/sharding.py`) consumed by a rewritten `adapter/run_adapter.py` (batching + bisection + atomic writes + sidecar telemetry). Standalone scripts (`scripts/`) for env-check, bootstrap, repro-verification, and shard-merge. CI on CPU runners; one manual GPU sanity run at the end.

**Tech Stack:** Python 3.11/3.12, vLLM 0.22.1 (ROCm, `vllm-0221b` venv), `omnidocbench-rocm==0.3.2`, pytest, ruff, `python -m build`, hatchling.

## Global Constraints

- **NO `git commit` / `git push` / PRs.** The user has not authorized commits. Every task ends at a **verification checkpoint** (command + expected result); leave changes in the working tree. Suggested commit messages are documentation only.
- **Do not modify any file under `results/`** (committed evidence + conformance requires `windows-hip/_run_stats.json` to keep that dir non-empty).
- **Do not recompute or alter the published 95.88** (model_card.overall must still equal `recompute_overall(metric)`).
- **Every behavior change has a test.** TDD: write failing test → implement → green.
- **Conformance invariants must hold:** both READMEs contain the literal section words `Install`, `Demo`, `Evaluation`, `Reproducibility`, `Known Gaps`; per-run accounting keeps `ok+fail+fallback == count` (so skipped pages keep `status="ok"` with `seconds=None`); `windows-hip/_run_stats.json` stays.
- **Backend is a closed set `{smoke, vllm}`.** `windows-hip + vllm` → explicit error, no silent fallback.
- **Canonical venv:** `/root/venvs/vllm-0221b`, overridable via `VENV=`, used identically in `00-install-deps.sh`, README, README.zh-CN, reproduce.md, REPRO.yaml.
- **Engine pin:** `omnidocbench-rocm==0.3.2` (verify CPU-installable in Task 6 before committing the pin).
- **CPU tests only** use a fake `vllm` module (opt-in fixture); never require a GPU or the 1.7 GB weights.
- Keep diffs small, focused, no unrelated refactors. Match existing code style.

---

## File Structure

**New files:**
- `adapter/postprocess.py` — `clean_truncated_repeats`, `postprocess` (moved out of `run_adapter.py` for testability).
- `adapter/sharding.py` — `select_shard`, `shard_dir`, `validate_shard_args`, `merge_shards` (shared by adapter + merge CLI).
- `scripts/merge-shards.py` — CLI over `adapter.sharding.merge_shards`.
- `scripts/check-environment.py` — verifies the active Python's stack; `--cpu-only` for CI.
- `scripts/bootstrap-linux.sh` — idempotent, pinned installer clone + weight pinning.
- `scripts/verify-reproduction-inputs.py` — checks REPRO.yaml (weights sha256 = `model.safetensors`, revisions, engine version).
- `tests/conftest.py` — opt-in `fake_vllm` fixture + shared helpers.
- `tests/test_config.py`, `tests/test_postprocess.py`, `tests/test_sharding.py`, `tests/test_adapter_smoke.py`, `tests/test_adapter_failures.py`, `tests/test_merge_shards.py`, `tests/test_cli.py`, `tests/test_repro_inputs.py`.
- `analysis/formula-gap/{README.md,compare.py,sample_manifest.example.json,.gitkeep}`.

**Modified files:**
- `adapter/adapter_config.py` — `Config` dataclass + `resolve()` + `validate()` + env loading.
- `adapter/run_adapter.py` — consume resolver/helpers; batching; atomic writes; timing; sidecars; collision guard; shard selection; CLI flags.
- `adapter/setup/00-install-deps.sh` — venv `vllm-0221b`, clone-not-curl, `VENV=` override.
- `Makefile` — `install-dev`/`check`/`smoke-test`/`demo-smoke`/`demo-real`/`eval-linux`(vllm)/`conformance`/`build`/`clean`; `eval-windows` fails clearly.
- `pyproject.toml` — build-system, metadata, `==0.3.2`, dev deps, pytest/ruff config.
- `.github/workflows/ci.yml` — matrix, real checks, permissions, timeout.
- `README.md`, `README.zh-CN.md`, `reproduce.md`, `REPRO.yaml`, `model_card.json` — see Task 8.

---

## Task 1: Config foundation (P0-2 core)

**Files:**
- Modify: `adapter/adapter_config.py` (full rewrite)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `adapter_config.Config` (frozen dataclass); `adapter_config.resolve(cli: dict) -> Config`; `adapter_config.validate(cfg: Config) -> None` (raises `SystemExit` on invalid); `adapter_config.load_env_local(path=None) -> None`; `adapter_config.config_snapshot(cfg) -> dict`.

- [ ] **Step 1: Write the failing test** — `tests/test_config.py`

```python
import os, sys
from pathlib import Path
import pytest

def _reload(monkeypatch):
    import importlib
    import adapter.adapter_config as m
    return importlib.reload(m)

def test_defaults_resolve_to_vllm(tmp_path, monkeypatch):
    monkeypatch.delenv("OVISOCR2_BACKEND", raising=False)
    m = _reload(monkeypatch)
    cfg = m.resolve({"platform": "linux-rocm"})
    assert cfg.backend == "vllm"
    assert cfg.temperature == 0.0
    assert cfg.batch_size == 8
    assert cfg.num_shards == 1 and cfg.shard_index == 0

def test_env_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")
    monkeypatch.setenv("OVISOCR2_BATCH_SIZE", "4")
    m = _reload(monkeypatch)
    cfg = m.resolve({"platform": "linux-rocm"})
    assert cfg.backend == "smoke" and cfg.batch_size == 4

def test_cli_explicit_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")
    m = _reload(monkeypatch)
    cfg = m.resolve({"platform": "linux-rocm", "backend": "vllm"})
    assert cfg.backend == "vllm"

def test_cli_none_does_not_clobber_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")
    m = _reload(monkeypatch)
    cfg = m.resolve({"platform": "linux-rocm", "backend": None})
    assert cfg.backend == "smoke"

def test_env_local_loaded_shell_wins(tmp_path, monkeypatch):
    env_local = tmp_path / ".env.local"
    env_local.write_text('OVISOCR2_BACKEND=vllm\nOVISOCR2_MAX_TOKENS=999\n')
    monkeypatch.setenv("OVISOCR2_BACKEND", "smoke")  # shell already set
    m = _reload(monkeypatch)
    m.load_env_local(env_local)
    cfg = m.resolve({"platform": "linux-rocm"})
    assert cfg.backend == "smoke"            # shell wins over .env.local
    assert cfg.max_tokens == 999             # .env.local fills unset keys

def test_invalid_backend_rejected(tmp_path, monkeypatch):
    m = _reload(monkeypatch)
    with pytest.raises(SystemExit):
        m.resolve({"platform": "linux-rocm", "backend": "bogus"})

def test_windows_hip_plus_vllm_rejected(tmp_path, monkeypatch):
    m = _reload(monkeypatch)
    with pytest.raises(SystemExit):
        m.resolve({"platform": "windows-hip", "backend": "vllm"})

def test_bad_shard_args_rejected(tmp_path, monkeypatch):
    m = _reload(monkeypatch)
    with pytest.raises(SystemExit):
        m.resolve({"platform": "linux-rocm", "num_shards": 2, "shard_index": 2})
    with pytest.raises(SystemExit):
        m.resolve({"platform": "linux-rocm", "num_shards": 0})
```

- [ ] **Step 2: Run test to verify it fails** — `Run: pytest tests/test_config.py -q` → FAIL (module lacks `resolve`).

- [ ] **Step 3: Implement** — replace `adapter/adapter_config.py` with:

```python
"""Adapter configuration for OvisOCR2-ROCm.

Priority: explicit CLI > environment (incl. adapter/setup/.env.local) > defaults.
CLI flags that override config arrive as ``None`` when unset, so an absent flag
never clobbers an env value. Backend is a closed set {smoke, vllm};
windows-hip + vllm is rejected with an explicit error (no silent fallback).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

VALID_BACKENDS = {"smoke", "vllm"}
VALID_PLATFORMS = {"linux-rocm", "windows-hip"}
ENV_LOCAL = Path(__file__).resolve().parent / "setup" / ".env.local"


@dataclass(frozen=True)
class Config:
    platform: str
    backend: str
    server_url: str = ""
    api_model_name: str = "ovisocr2"
    weights_dir: str = ""
    max_tokens: int = 16384
    temperature: float = 0.0
    min_pixels: int = 448 * 448
    max_pixels: int = 2880 * 2880
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = True
    trust_remote_code: bool = True
    gdn_prefill_backend: str = "triton"
    batch_size: int = 8
    num_shards: int = 1
    shard_index: int = 0
    skip_existing: bool = False
    limit_pages: int | None = None


def load_env_local(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from .env.local. Shell (os.environ) always wins."""
    p = Path(path) if path else ENV_LOCAL
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _pick(cli_val, env_name, default, cast=str):
    if cli_val is not None:
        return cli_val
    if env_name and env_name in os.environ:
        return cast(os.environ[env_name])
    return default


def _bool(s):  return str(s).strip().lower() in ("1", "true", "yes", "on")


def resolve(cli: dict) -> Config:
    """Build a validated Config. ``cli`` values are None when the flag was unset."""
    load_env_local()
    cfg = Config(
        platform=cli.get("platform") or os.environ.get("OVISOCR2_PLATFORM", "linux-rocm"),
        backend=_pick(cli.get("backend"), "OVISOCR2_BACKEND", "vllm"),
        server_url=_pick(cli.get("server_url"), "OVISOCR2_SERVER_URL", ""),
        api_model_name=_pick(cli.get("api_model_name"), "OVISOCR2_API_MODEL_NAME", "ovisocr2"),
        weights_dir=_pick(cli.get("weights_dir"), "OVISOCR2_WEIGHTS", ""),
        max_tokens=_pick(cli.get("max_tokens"), "OVISOCR2_MAX_TOKENS", 16384, int),
        temperature=_pick(cli.get("temperature"), "OVISOCR2_TEMPERATURE", 0.0, float),
        min_pixels=_pick(cli.get("min_pixels"), "OVISOCR2_MIN_PIXELS", 448 * 448, int),
        max_pixels=_pick(cli.get("max_pixels"), "OVISOCR2_MAX_PIXELS", 2880 * 2880, int),
        max_model_len=_pick(cli.get("max_model_len"), "OVISOCR2_MAX_MODEL_LEN", 32768, int),
        gpu_memory_utilization=_pick(cli.get("gpu_memory_utilization"), "OVISOCR2_GPU_MEM_UTIL", 0.9, float),
        enforce_eager=_pick(cli.get("enforce_eager"), "OVISOCR2_ENFORCE_EAGER", True, _bool),
        trust_remote_code=_pick(cli.get("trust_remote_code"), "OVISOCR2_TRUST_REMOTE_CODE", True, _bool),
        gdn_prefill_backend=_pick(cli.get("gdn_prefill_backend"), "OVISOCR2_GDN_PREFILL_BACKEND", "triton"),
        batch_size=_pick(cli.get("batch_size"), "OVISOCR2_BATCH_SIZE", 8, int),
        num_shards=_pick(cli.get("num_shards"), "OVISOCR2_NUM_SHARDS", 1, int),
        shard_index=_pick(cli.get("shard_index"), "OVISOCR2_SHARD_INDEX", 0, int),
        skip_existing=_pick(cli.get("skip_existing"), "OVISOCR2_SKIP_EXISTING", False, _bool),
        limit_pages=cli.get("limit_pages"),
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.backend not in VALID_BACKENDS:
        raise SystemExit(f"invalid backend {cfg.backend!r}; allowed: {sorted(VALID_BACKENDS)}")
    if cfg.platform not in VALID_PLATFORMS:
        raise SystemExit(f"invalid platform {cfg.platform!r}; allowed: {sorted(VALID_PLATFORMS)}")
    if cfg.platform == "windows-hip" and cfg.backend == "vllm":
        raise SystemExit(
            "windows-hip + vllm is unsupported (Qwen3-Next GDN has no HIP-SDK serving path; "
            "community-wanted). Refusing to fall back to smoke.")
    if cfg.batch_size < 1:
        raise SystemExit(f"batch_size must be >= 1 (got {cfg.batch_size})")
    if cfg.num_shards < 1:
        raise SystemExit(f"num_shards must be >= 1 (got {cfg.num_shards})")
    if not (0 <= cfg.shard_index < cfg.num_shards):
        raise SystemExit(f"need 0 <= shard_index < num_shards (got {cfg.shard_index}/{cfg.num_shards})")


def config_snapshot(cfg: Config) -> dict:
    """Public, JSON-serializable view for run_stats/sidecar provenance."""
    return asdict(cfg)
```

- [ ] **Step 4: Run test to verify it passes** — `Run: pytest tests/test_config.py -q` → PASS (8 tests).

- [ ] **Step 5: Checkpoint (no commit)** — `ruff check adapter/adapter_config.py tests/test_config.py` clean. Suggested message: `feat(config): dataclass Config + CLI>env>.env.local>defaults resolver, backend whitelist, windows-hip guard`.

---

## Task 2: Pure helpers — postprocess + sharding primitives

**Files:**
- Create: `adapter/postprocess.py`
- Create: `adapter/sharding.py`
- Test: `tests/test_postprocess.py`, `tests/test_sharding.py`

**Interfaces:**
- `postprocess.clean_truncated_repeats(text, ...) -> str`; `postprocess.postprocess(text) -> str`.
- `sharding.validate_shard_args(num_shards, shard_index) -> None`; `sharding.select_shard(images: list[Path], num_shards, shard_index) -> list[Path]`; `sharding.shard_dir(out_dir: Path, num_shards, shard_index) -> Path`; `sharding.merge_shards(input_root: Path, expected_images: list[Path]) -> MergeReport`.

- [ ] **Step 1: Write failing tests**

`tests/test_postprocess.py`:
```python
from adapter import postprocess as P

def test_drops_visual_region_img_tags():
    src = '<img src="images/bbox_1_2_3_4.jpg" />\n\nreal text'
    assert P.postprocess(src).strip() == "real text"

def test_clean_truncated_repeats_short_text_untouched():
    assert P.clean_truncated_repeats("short") == "short"

def test_clean_truncated_repeats_trims_tail(monkeypatch):
    # force the guard low so a small crafted input triggers trimming
    tail = ("abcd" * 50)  # 200 chars, repeat unit 4, >5 times, >100 chars
    text = "HEAD" + tail
    out = P.clean_truncated_repeats(text, min_text_len=10, min_repeat_chars=20, min_repeat_times=5)
    assert "HEAD" in out and len(out) < len(text)

def test_utf8_roundtrip():
    s = "中文公式 $x = \\frac{1}{2}$ — ✓"
    assert P.postprocess(s) == s
```

`tests/test_sharding.py`:
```python
from pathlib import Path
from adapter import sharding as S

def imgs(n): return [Path(f"page_{i:03d}.png") for i in range(n)]

def test_two_shards_disjoint_and_cover():
    a = S.select_shard(imgs(10), 2, 0); b = S.select_shard(imgs(10), 2, 1)
    assert set(a) | set(b) == set(imgs(10)) and not (set(a) & set(b))

@pytest.mark.parametrize("n", [3, 4, 5])
def test_n_shards_cover_all(n):
    parts = [S.select_shard(imgs(10), n, i) for i in range(n)]
    flat = [p for part in parts for p in part]
    assert len(flat) == 10 and set(flat) == set(imgs(10))

def test_deterministic_and_sorted():
    assert S.select_shard(imgs(10), 3, 1) == sorted(imgs(10))[1::3]

def test_bad_args_rejected():
    import pytest
    with pytest.raises(ValueError): S.select_shard(imgs(5), 0, 0)
    with pytest.raises(ValueError): S.select_shard(imgs(5), 2, 2)
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_postprocess.py tests/test_sharding.py -q` → FAIL (modules missing).

- [ ] **Step 3: Implement `adapter/postprocess.py`**

```python
"""OvisOCR2 post-processing (upstream model card recipe, verbatim)."""
from __future__ import annotations


def clean_truncated_repeats(text, min_text_len=8000, max_period=200, min_period=1,
                            min_repeat_chars=100, min_repeat_times=5):
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len, idx = 1, n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1; idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]
    return text


def postprocess(text: str) -> str:
    """Drop visual-region <img> tags (card default) then clean repeat tails."""
    text = "\n\n".join(
        b for b in text.split("\n\n")
        if not b.strip().startswith('<img src="images/bbox_'))
    return clean_truncated_repeats(text)
```

- [ ] **Step 4: Implement `adapter/sharding.py`**

```python
"""Deterministic sharding + shard-merge validation (shared by adapter + CLI)."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1 (got {num_shards})")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"need 0 <= shard_index < num_shards (got {shard_index}/{num_shards})")


def select_shard(images: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    validate_shard_args(num_shards, shard_index)
    ordered = sorted(images)
    return ordered[shard_index::num_shards]


def shard_dir(out_dir: Path, num_shards: int, shard_index: int) -> Path:
    """Multi-shard runs write to a per-shard subdir; single-shard writes to out_dir."""
    if num_shards == 1:
        return Path(out_dir)
    return Path(out_dir) / f"shard-{shard_index:05d}-of-{num_shards:05d}"


@dataclass
class MergeReport:
    ok: bool = True
    merged_dir: Path | None = None
    page_count: int = 0
    missing: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    failed_pages: list[str] = field(default_factory=list)
    shard_configs: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str):
        self.errors.append(msg); self.ok = False


def merge_shards(input_root: Path, expected_images: list[Path], out_dir: Path) -> MergeReport:
    """Merge per-shard outputs into out_dir. Never last-write-wins on conflict."""
    input_root = Path(input_root); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = MergeReport(merged_dir=out_dir)
    shard_dirs = sorted(p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("shard-"))
    if not shard_dirs:  # fall back to single-dir layout
        shard_dirs = [input_root]
    expected_stems = sorted(p.stem for p in expected_images)
    seen: dict[str, Path] = {}
    for sd in shard_dirs:
        stats_p = sd / "_run_stats.json"
        if stats_p.exists():
            try:
                rs = json.loads(stats_p.read_text(encoding="utf-8"))
                rep.shard_configs.append({k: rs.get(k) for k in ("engine", "count", "ok", "fail", "fallback")})
                for pg in rs.get("stats", []):
                    if str(pg.get("status", "")).startswith("failed"):
                        rep.failed_pages.append(pg.get("image", "?"))
            except Exception as e:
                rep.fail(f"unreadable {stats_p}: {e}")
        for md in sorted(sd.glob("*.md")):
            stem = md.stem
            content = md.read_bytes()
            if stem in seen:
                if seen[stem].read_bytes() != content:
                    rep.conflicts.append(stem)
                else:
                    rep.duplicates.append(stem)
                continue
            seen[stem] = md
            target = out_dir / md.name
            tmp = target.with_suffix(".md.tmp")
            tmp.write_bytes(content); tmp.replace(target)
            rep.page_count += 1
    # coverage checks
    got = sorted(seen)
    rep.missing = sorted(set(expected_stems) - set(got))
    extra = sorted(set(got) - set(expected_stems))
    if rep.missing:
        rep.fail(f"missing pages: {rep.missing[:10]}{' ...' if len(rep.missing) > 10 else ''}")
    if extra:
        rep.fail(f"unexpected pages not in expected set: {extra[:10]}")
    if rep.conflicts:
        rep.fail(f"content conflicts (different bytes, same stem): {rep.conflicts}")
    if rep.failed_pages:
        rep.fail(f"failed pages present in shards: {rep.failed_pages[:10]}")
    return rep
```

- [ ] **Step 5: Run tests to verify they pass** — `pytest tests/test_postprocess.py tests/test_sharding.py -q` → PASS.

- [ ] **Step 5b: Add `tests/conftest.py`** (opt-in fake vllm for later tasks):

```python
import sys, types
import pytest

class _FakeOutput: 
    def __init__(self, text): self.text = text
class _FakeRequestOutput:
    def __init__(self, text): self.outputs = [_FakeOutput(text)]

class FakeLLM:
    """Per-page scripted outcomes. outcomes: dict[stem->str|Exception]."""
    def __init__(self, outcomes=None, batch_fail=False):
        self._outcomes = outcomes or {}; self._batch_fail = batch_fail
    def generate(self, inputs, sp):
        if self._batch_fail:
            raise RuntimeError("simulated batch OOM")
        out = []
        for inp in inputs:
            img = inp["multi_modal_data"]["image"]
            stem = getattr(img, "stem", None) or "x"
            o = self._outcomes.get(stem, f"# {stem}\n\nbody")
            if isinstance(o, Exception): raise o
            out.append(_FakeRequestOutput(o))
        return out
    def get_tokenizer(self):
        class _T:
            def apply_chat_template(self, msgs, **kw):
                return "<|prompt|>"
        return _T()

class _FakeSamplingParams:
    def __init__(self, *a, **k): pass

@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a fake vllm module so the vllm branch runs on CPU."""
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    mod.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod
```

- [ ] **Step 6: Checkpoint (no commit)** — `ruff check adapter/postprocess.py adapter/sharding.py tests/`. Suggested message: `feat(adapter): extract tested postprocess + deterministic sharding primitives + fake_vllm fixture`.

---

## Task 3: Rewrite `run_adapter.py` — config wiring, batching, recovery, atomic writes, telemetry (P0-2/3/4/6)

**Files:**
- Modify: `adapter/run_adapter.py` (full rewrite of inference + CLI)
- Test: `tests/test_adapter_smoke.py`, `tests/test_adapter_failures.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `adapter_config.resolve`, `adapter_config.config_snapshot`; `postprocess.postprocess`; `sharding.select_shard`, `sharding.shard_dir`.
- Produces: `run_adapter.run_adapter(img_dir, out_dir, *, platform, config) -> dict` (unchanged signature); writes `out_dir/<stem>.md`, `_run_stats.json`, and (vllm only) `_pages.jsonl` + `_performance.json`.

- [ ] **Step 1: Write failing tests**

`tests/test_adapter_smoke.py` (smoke path, no GPU, no fake needed):
```python
import json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "adapter" / "run_adapter.py"
DEMO = REPO / "examples" / "demo.png"

def _run(args, **kw):
    return subprocess.run([sys.executable, str(ADAPTER), *args],
                          capture_output=True, text=True, **kw)

def test_smoke_writes_contract_outputs(tmp_path):
    img = tmp_path / "imgs"; img.mkdir()
    (img / "page.png").write_bytes(DEMO.read_bytes())
    out = tmp_path / "out"
    r = _run(["--img-dir", str(img), "--out-dir", str(out),
              "--platform", "linux-rocm", "--backend", "smoke"])
    assert r.returncode == 0, r.stderr
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["schema_version"] == 1 and rs["count"] == 1 and rs["ok"] == 1 and rs["engine"] == "smoke"
    assert (out / "page.md").read_text(encoding="utf-8").strip()

def test_smoke_empty_dir_is_ok(tmp_path):
    img = tmp_path / "empty"; img.mkdir(); out = tmp_path / "out"
    r = _run(["--img-dir", str(img), "--out-dir", str(out),
              "--platform", "linux-rocm", "--backend", "smoke"])
    assert r.returncode == 0
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 0

def test_smoke_does_not_import_vllm(tmp_path):
    img = tmp_path / "imgs"; img.mkdir(); (img / "page.png").write_bytes(DEMO.read_bytes())
    probe = 'import sys,runpy; '
    probe += 'sys.argv=["x","--img-dir","'+str(img)+'","--out-dir","'+str(tmp_path/'out')+'","--platform","linux-rocm","--backend","smoke"]; '
    probe += 'runpy.run_path("' + str(ADAPTER) + '", run_name="__main__"); '
    probe += 'assert "vllm" not in sys.modules, "smoke imported vllm"'
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

`tests/test_adapter_failures.py` (vllm path via fake_vllm):
```python
import json
from pathlib import Path
from PIL import Image
import adapter.run_adapter as R

def _make_imgs(tmp_path, names):
    d = tmp_path / "imgs"; d.mkdir()
    for n in names:
        Image.new("RGB", (8, 8), "white").save(d / n)
    return d

def test_bad_image_does_not_abort_run(tmp_path, fake_vllm, monkeypatch):
    d = _make_imgs(tmp_path, ["good.png"])
    (d / "bad.png").write_bytes(b"not an image")
    out = tmp_path / "out"
    fake_vllm.LLM = lambda *a, **k: __import__("tests.conftest", fromlist=["FakeLLM"]).FakeLLM()
    R.run_adapter(d, out, platform="linux-rocm",
                  config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "fake"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 2 and rs["ok"] == 1 and rs["fail"] == 1

def test_empty_output_recorded_as_failed(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["p.png"]); out = tmp_path / "out"
    from tests.conftest import FakeLLM
    fake_vllm.LLM = lambda *a, **k: FakeLLM(outcomes={"p": "   "})
    R.run_adapter(d, out, platform="linux-rocm",
                  config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "fake"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["fail"] == 1 and rs["ok"] == 0

def test_skip_existing_excluded_from_throughput(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["a.png", "b.png"]); out = tmp_path / "out"; out.mkdir()
    (out / "a.md").write_text("existing")   # pre-existing -> skipped
    from tests.conftest import FakeLLM
    fake_vllm.LLM = lambda *a, **k: FakeLLM()
    R.run_adapter(d, out, platform="linux-rocm",
                  config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "fake",
                          "skip_existing": True})
    rs = json.loads((out / "_run_stats.json").read_text())
    # ok count includes skipped (conformance: ok+fail+fallback==count), but skipped seconds is null
    assert rs["ok"] == 2
    skipped = [s for s in rs["stats"] if s["image"] == "a.png"][0]
    assert skipped["seconds"] is None and skipped["attempts"] == 0
    # sidecar distinguishes generated vs skipped
    perf = json.loads((out / "_performance.json").read_text())
    assert perf["generated_pages"] == 1 and perf["skipped_pages"] == 1

def test_stem_collision_rejected(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["page.png"])
    (d / "page.jpg").write_bytes(b"\xff\xd8")  # same stem, different ext
    import pytest
    with pytest.raises(SystemExit):
        R.run_adapter(d, tmp_path / "out", platform="linux-rocm",
                      config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "fake"})

def test_atomic_write(tmp_path, fake_vllm):
    d = _make_imgs(tmp_path, ["p.png"]); out = tmp_path / "out"
    from tests.conftest import FakeLLM
    fake_vllm.LLM = lambda *a, **k: FakeLLM(outcomes={"p": "# p\n\nhello"})
    R.run_adapter(d, out, platform="linux-rocm",
                  config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "fake"})
    assert (out / "p.md").read_text() == "# p\n\nhello"
    assert not (out / "p.md.tmp").exists()  # tmp cleaned up

def test_batch_failure_localizes_page(tmp_path, fake_vllm):
    # whole-batch generate raises -> bisection -> each page recorded failed
    d = _make_imgs(tmp_path, ["a.png", "b.png"]); out = tmp_path / "out"
    from tests.conftest import FakeLLM
    fake_vllm.LLM = lambda *a, **k: FakeLLM(batch_fail=True)
    R.run_adapter(d, out, platform="linux-rocm",
                  config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "fake",
                          "batch_size": 8})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["fail"] == 2 and rs["ok"] == 0
```

`tests/test_cli.py`:
```python
import subprocess, sys
from pathlib import Path
ADAPTER = Path(__file__).resolve().parents[1] / "adapter" / "run_adapter.py"

def _run(args): return subprocess.run([sys.executable, str(ADAPTER), *args], capture_output=True, text=True)

def test_help_exits_zero():
    r = _run(["--help"]); assert r.returncode == 0 and "--batch-size" in r.stdout and "--num-shards" in r.stdout

def test_invalid_backend_exits_nonzero(tmp_path):
    r = _run(["--img-dir", str(tmp_path), "--out-dir", str(tmp_path/"o"),
              "--platform", "linux-rocm", "--backend", "bogus"])
    assert r.returncode != 0 and "backend" in r.stderr.lower()

def test_windows_hip_vllm_rejected(tmp_path):
    r = _run(["--img-dir", str(tmp_path), "--out-dir", str(tmp_path/"o"),
              "--platform", "windows-hip", "--backend", "vllm"])
    assert r.returncode != 0 and "windows-hip" in r.stderr
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_adapter_smoke.py tests/test_adapter_failures.py tests/test_cli.py -q` → FAIL.

- [ ] **Step 3: Implement** — replace `adapter/run_adapter.py` with:

```python
"""OvisOCR2-ROCm adapter — implements the omnidocbench-rocm contract.

Contract notes
--------------
* Engine invokes this as a subprocess and consumes only
  out_dir/<image_stem>.md + _run_stats.json (R1).
* Per-page failures are caught and recorded; the run never raises (R2),
  except a hard config error (bad backend, stem collision) raised up front.
* One UTF-8 .md per page image, named by stem (R3).
* backend == "smoke" is a no-GPU placeholder (CI gate).
* Multi-shard runs write per-shard subdirs (shard-NNNNN-of-NNNNN/).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

from omnidocbench_rocm.types import PageStatus, RunSummary

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")

_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    "represent them using an HTML image tag: <"
    'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)


def _load_helpers():
    try:
        from . import adapter_config, postprocess, sharding
    except ImportError:
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        import adapter_config, postprocess, sharding  # type: ignore
    return adapter_config, postprocess, sharding


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with open(tmp, "ab") as f:
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ---- vLLM (lazy) -----------------------------------------------------------
_LLM = None
_CHAT = None


def _get_llm(cfg):
    global _LLM, _CHAT
    if _LLM is not None:
        return _LLM, _CHAT
    from vllm import LLM
    weights = cfg.weights_dir or os.environ.get("OVISOCR2_WEIGHTS") or "ATH-MaaS/OvisOCR2"
    kwargs = dict(model=weights, tensor_parallel_size=1,
                  gpu_memory_utilization=cfg.gpu_memory_utilization,
                  max_model_len=cfg.max_model_len, trust_remote_code=cfg.trust_remote_code,
                  enforce_eager=cfg.enforce_eager, limit_mm_per_prompt={"image": 1})
    try:
        _LLM = LLM(gdn_prefill_backend=cfg.gdn_prefill_backend, **kwargs)
    except TypeError:
        _LLM = LLM(**kwargs)
    _CHAT = _LLM.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return _LLM, _CHAT


def _build_input(chat, image_path, min_px, max_px):
    from PIL import Image
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    return {"prompt": chat, "multi_modal_data": {"image": rgb},
            "mm_processor_kwargs": {"images_kwargs": {"min_pixels": min_px, "max_pixels": max_px}}}


def _run_batch(llm, chat, sp, postprocess_fn, images, min_px, max_px):
    """Try a batch; on a whole-batch failure, bisect to single pages.
    Returns list[(image, text|None, error|None)]. Never raises."""
    try:
        inputs = [_build_input(chat, i, min_px, max_px) for i in images]
        t0 = time.time()
        outputs = llm.generate(inputs, sp)
        wall = time.time() - t0
    except Exception as e:  # whole-batch failure (OOM, generate error, bad image in build)
        if len(images) == 1:
            return [(images[0], None, wall_or_none(), f"batch failed: {e}")]
        mid = len(images) // 2
        return (_run_batch(llm, chat, sp, postprocess_fn, images[:mid], min_px, max_px)
                + _run_batch(llm, chat, sp, postprocess_fn, images[mid:], min_px, max_px))
    # batch succeeded -> per-page postprocess (still isolated)
    share = wall / len(images) if images else 0.0
    results = []
    for i, out in zip(images, outputs):
        try:
            text = postprocess_fn(out.outputs[0].text.strip())
            if not text.strip():
                results.append((i, None, None, "empty prediction"))
            else:
                results.append((i, text, share if len(images) == 1 else None, None))
        except Exception as e:
            results.append((i, None, None, str(e)))
    return results
```
(Note: `_run_batch` returns 4-tuples `(image, text, seconds, error)`; the `wall_or_none()` placeholder above must be replaced — see correction in Step 3b.)

- [ ] **Step 3b: Correct `_run_batch` 4-tuple shape** — replace the `len==1` branch's `wall_or_none()` with a real measurement. Final `_run_batch`:

```python
def _run_batch(llm, chat, sp, postprocess_fn, images, min_px, max_px):
    try:
        inputs = [_build_input(chat, i, min_px, max_px) for i in images]
        t0 = time.time()
        outputs = llm.generate(inputs, sp)
        wall = time.time() - t0
    except Exception as e:
        if len(images) == 1:
            return [(images[0], None, None, f"batch failed: {e}")]
        mid = len(images) // 2
        return (_run_batch(llm, chat, sp, postprocess_fn, images[:mid], min_px, max_px)
                + _run_batch(llm, chat, sp, postprocess_fn, images[mid:], min_px, max_px))
    share = wall / len(images) if images else 0.0
    out_list = []
    for i, out in zip(images, outputs):
        try:
            text = postprocess_fn(out.outputs[0].text.strip())
            if not text.strip():
                out_list.append((i, None, None, "empty prediction"))
            else:
                out_list.append((i, text, share if len(images) == 1 else None, None))
        except Exception as e:
            out_list.append((i, None, None, str(e)))
    return out_list
```

- [ ] **Step 3c: Implement `run_adapter()` + `main()`** (append to the file):

```python
def run_adapter(img_dir, out_dir, *, platform, config) -> dict:
    adapter_config, postprocess, sharding = _load_helpers()
    cli = {**config, "platform": platform}
    cfg = adapter_config.resolve(cli)
    out_dir = Path(out_dir)
    target_dir = sharding.shard_dir(out_dir, cfg.num_shards, cfg.shard_index)
    target_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    if cfg.limit_pages is not None:
        all_imgs = all_imgs[: cfg.limit_pages]
    imgs = sharding.select_shard(all_imgs, cfg.num_shards, cfg.shard_index)

    # Stem-collision precheck (hard error before any inference).
    seen = {}
    for i in imgs:
        if i.stem in seen:
            raise SystemExit(f"stem collision: {i.name} and {seen[i.stem].name} both map to {i.stem}.md; refusing to overwrite")
        seen[i.stem] = i

    stats: list[PageStatus] = []
    pages_jsonl = []  # rich per-page sidecar rows
    generated = skipped = failed = 0
    infer_wall = 0.0
    started = datetime.now(timezone.utc)

    if cfg.backend == "smoke":
        if not imgs and platform != "linux-rocm":
            pass  # empty dir is fine for smoke
        for i in imgs:
            _atomic_write_text(target_dir / f"{i.stem}.md", f"# {i.stem}\n\n(smoke output — backend=smoke)\n")
            stats.append(PageStatus(i.name, "ok", seconds=None, attempts=0))
            pages_jsonl.append({"image": i.name, "status": "ok", "shard_index": cfg.shard_index})
    else:
        from vllm import SamplingParams  # lazy
        llm, chat = _get_llm(cfg)
        load_done = datetime.now(timezone.utc)
        model_load_seconds = (load_done - started).total_seconds()
        sp = SamplingParams(max_tokens=cfg.max_tokens, temperature=cfg.temperature)
        todo = []
        for i in imgs:
            target = target_dir / f"{i.stem}.md"
            if cfg.skip_existing and target.exists() and target.read_text(encoding="utf-8").strip():
                stats.append(PageStatus(i.name, "ok", seconds=None, attempts=0))  # skipped: status ok, seconds null
                pages_jsonl.append({"image": i.name, "status": "skipped", "shard_index": cfg.shard_index})
                skipped += 1
            else:
                todo.append(i)
        for start in range(0, len(todo), cfg.batch_size):
            batch = todo[start:start + cfg.batch_size]
            bt0 = time.time()
            for img, text, seconds, error in _run_batch(llm, chat, sp, postprocess.postprocess,
                                                         batch, cfg.min_pixels, cfg.max_pixels):
                if error is not None:
                    stats.append(PageStatus(img.name, f"failed: {error}", error=error))
                    pages_jsonl.append({"image": img.name, "status": "failed", "shard_index": cfg.shard_index,
                                        "error": error})
                    failed += 1
                else:
                    _atomic_write_text(target_dir / f"{img.stem}.md", text)
                    nbytes = len(text.encode("utf-8"))
                    stats.append(PageStatus(img.name, "ok", seconds=seconds, attempts=1))
                    pages_jsonl.append({"image": img.name, "status": "generated", "shard_index": cfg.shard_index,
                                        "seconds": seconds, "output_bytes": nbytes})
                    generated += 1
            infer_wall += time.time() - bt0
            done = start + len(batch)
            print(f"[ovisocr2] batch {start//cfg.batch_size + 1}: {done}/{len(todo)} done "
                  f"(skipped {skipped}, failed {failed})", file=sys.stderr)
        _write_sidecars(target_dir, cfg, started, model_load_seconds, infer_wall,
                        generated, skipped, failed, platform, pages_jsonl)

    rs = RunSummary(
        len(imgs),
        sum(1 for s in stats if s.status == "ok"),
        sum(1 for s in stats if s.status.startswith("failed")),
        sum(1 for s in stats if s.status.startswith("fallback")),
        cfg.limit_pages, stats, engine=cfg.backend)
    rs.write(target_dir / "_run_stats.json")
    return rs.to_run_stats()


def _write_sidecars(target_dir, cfg, started, model_load_seconds, infer_wall,
                    generated, skipped, failed, platform, pages_jsonl):
    import subprocess
    completed = datetime.now(timezone.utc)
    total = (completed - started).total_seconds()
    env = _collect_env()  # best-effort versions/git/gpu; None if unavailable
    perf = {
        "schema_version": 2, "started_at_utc": started.isoformat(), "completed_at_utc": completed.isoformat(),
        "model_load_seconds": round(model_load_seconds, 3), "inference_wall_seconds": round(infer_wall, 3),
        "total_wall_seconds": round(total, 3), "generated_pages": generated, "skipped_pages": skipped,
        "failed_pages": failed, "limit_pages": cfg.limit_pages,
        "mean_latency_s_per_page": round(infer_wall / generated, 3) if generated else None,
        "backend": cfg.backend, "platform": platform, "num_shards": cfg.num_shards, "shard_index": cfg.shard_index,
        **env, "config_snapshot": _snapshot(cfg),
    }
    (target_dir / "_performance.json").write_text(json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(target_dir / "_pages.jsonl", "w", encoding="utf-8") as f:
        for row in pages_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _snapshot(cfg):
    from adapter import adapter_config
    return adapter_config.config_snapshot(cfg)


def _collect_env():
    out = {"output_tokens": None, "tokens_per_second": None,
           "max_memory_allocated_mb": None, "max_memory_reserved_mb": None,
           "gpu": None, "gfx_arch": None, "torch_version": None, "hip_version": None,
           "vllm_version": None, "transformers_version": None,
           "git_commit": None, "git_dirty": None, "weights_revision": None}
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["hip_version"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            out["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            out["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for modname, key in (("vllm", "vllm_version"), ("transformers", "transformers_version")):
        try:
            out[key] = __import__(modname).__version__
        except Exception:
            pass
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_repo_root())
        out["git_commit"] = rev.stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=_repo_root())
        out["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        pass
    return out


def _repo_root():
    return Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OvisOCR2-ROCm OmniDocBench adapter")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default=None)          # None -> resolve (default vllm)
    p.add_argument("--server-url", default=None)
    p.add_argument("--api-model-name", default=None)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--limit-pages", type=int, default=None)
    a = p.parse_args(argv)
    run_adapter(Path(a.img_dir), Path(a.out_dir), platform=a.platform, config={
        "backend": a.backend, "server_url": a.server_url, "api_model_name": a.api_model_name,
        "skip_existing": a.skip_existing, "batch_size": a.batch_size,
        "num_shards": a.num_shards, "shard_index": a.shard_index, "limit_pages": a.limit_pages,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_adapter_smoke.py tests/test_adapter_failures.py tests/test_cli.py -q` → PASS.

- [ ] **Step 5: Regression** — `pytest tests/test_smoke.py -q` → still PASS (smoke CLI passes `--backend smoke` explicitly; `_run_stats` shape unchanged). Update `tests/test_smoke.py` only if its `engine=="smoke"` assertion breaks (it won't).

- [ ] **Step 6: Checkpoint (no commit)** — `ruff check adapter/run_adapter.py tests/`. Suggested message: `feat(adapter): config-driven backend (no silent smoke), batching+bisection, atomic writes, honest timing, _performance.json/_pages.jsonl sidecars, shard selection, stem-collision guard`.

---

## Task 4: `scripts/merge-shards.py` (P0-4)

**Files:**
- Create: `scripts/merge-shards.py`
- Test: `tests/test_merge_shards.py`

**Interfaces:**
- Consumes: `sharding.merge_shards`, `sharding.IMG_EXT`.

- [ ] **Step 1: Write failing test** — `tests/test_merge_shards.py`:

```python
import json, subprocess, sys
from pathlib import Path
from adapter import sharding
import adapter.run_adapter as R

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "scripts" / "merge-shards.py"

def _imgs(tmp, names):
    from PIL import Image
    d = tmp / "imgs"; d.mkdir()
    for n in names: Image.new("RGB", (8, 8)).save(d / n)
    return d

def _shard_run(tmp, fake_vllm, names, num, idx):
    from tests.conftest import FakeLLM
    fake_vllm.LLM = lambda *a, **k: FakeLLM()
    d = _imgs(tmp, names)
    R.run_adapter(d, tmp / "preds", platform="linux-rocm",
                  config={"backend": "vllm", "platform": "linux-rocm", "weights_dir": "f",
                          "num_shards": num, "shard_index": idx})

def test_two_shards_merge_clean(tmp_path, fake_vllm):
    names = [f"p_{i:02d}.png" for i in range(6)]
    for idx in (0, 1):
        _shard_run(tmp_path, fake_vllm, names, 2, idx)  # writes into shard-*/ subdirs
    expected = sorted((tmp_path / "imgs").glob("*.png"))
    r = subprocess.run([sys.executable, str(MERGE), "--input-root", str(tmp_path/"preds"),
                        "--out-dir", str(tmp_path/"merged"), "--expected-images", str(tmp_path/"imgs")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    merged = sorted((tmp_path/"merged").glob("*.md"))
    assert len(merged) == 6
    assert [p.stem for p in merged] == sorted(p.stem for p in expected)

def test_merge_detects_missing_page(tmp_path, fake_vllm):
    names = [f"p_{i:02d}.png" for i in range(6)]
    _shard_run(tmp_path, fake_vllm, names, 2, 0)
    _shard_run(tmp_path, fake_vllm, names, 2, 1)
    # delete one expected so it's "missing"
    (tmp_path/"imgs"/"p_03.png").unlink()
    r = subprocess.run([sys.executable, str(MERGE), "--input-root", str(tmp_path/"preds"),
                        "--out-dir", str(tmp_path/"merged"), "--expected-images", str(tmp_path/"imgs")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "p_03" in r.stderr

def test_merge_detects_conflict(tmp_path, fake_vllm):
    names = [f"p_{i:02d}.png" for i in range(4)]
    _shard_run(tmp_path, fake_vllm, names, 2, 0)
    _shard_run(tmp_path, fake_vllm, names, 2, 1)
    # corrupt one shard's copy of a shared... (shards are disjoint, so craft a conflict)
    d = _imgs(tmp_path, ["x.png"]); d2 = tmp_path / "extra"; d2.mkdir()
    (tmp_path/"preds"/"shard-00000-of-00002"/"x.md").write_text("A")  # inject duplicate stem w/ diff content
    r = subprocess.run([sys.executable, str(MERGE), "--input-root", str(tmp_path/"preds"),
                        "--out-dir", str(tmp_path/"merged"), "--expected-images", str(tmp_path/"imgs")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "conflict" in r.stderr.lower()
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_merge_shards.py -q` → FAIL (no script).

- [ ] **Step 3: Implement `scripts/merge-shards.py`**:

```python
#!/usr/bin/env python3
"""Merge deterministic per-shard adapter outputs into one directory with validation.

Checks: no missing pages, no duplicates, no content conflicts (never last-write-wins),
no failed pages in shards, page count matches expected. Exits 1 on any problem.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from adapter.sharding import merge_shards, IMG_EXT  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge sharded adapter outputs with validation")
    p.add_argument("--input-root", required=True, help="dir holding shard-*/ subdirs (or a single output dir)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--expected-images", required=True, help="dir (or list) of expected input images")
    a = p.parse_args(argv)

    root = Path(a.expected_images)
    if root.is_dir():
        expected = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXT)
    else:
        expected = sorted(Path(x) for x in root.read_text().splitlines())

    rep = merge_shards(Path(a.input_root), expected, Path(a.out_dir))
    if rep.ok:
        print(f"[merge] OK: {rep.page_count} pages -> {a.out_dir}")
        return 0
    print("[merge] NON-CONFORMANT:", file=sys.stderr)
    for e in rep.errors:
        print(f"  - {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify pass** — `pytest tests/test_merge_shards.py -q` → PASS.

- [ ] **Step 5: Checkpoint (no commit)** — `python scripts/merge-shards.py --help` exits 0. Suggested message: `feat(scripts): merge-shards with coverage/duplicate/conflict validation`.

---

## Task 5: Environment, bootstrap, repro-verification scripts (P0-1)

**Files:**
- Create: `scripts/check-environment.py`, `scripts/bootstrap-linux.sh`, `scripts/verify-reproduction-inputs.py`
- Modify: `adapter/setup/00-install-deps.sh`
- Test: `tests/test_repro_inputs.py`

**Interfaces:** `check-environment.py` exits 0 only if the active Python has a qwen3_5-capable vLLM (+ torch/ROCm unless `--cpu-only`); `verify-reproduction-inputs.py` reads `REPRO.yaml` and checks weights sha256 (`model.safetensors`), dataset revision, engine version, git commit.

- [ ] **Step 1: Write failing test** — `tests/test_repro_inputs.py`:

```python
import subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]

def test_repro_inputs_help():
    r = subprocess.run([sys.executable, str(REPO/"scripts"/"verify-reproduction-inputs.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0

def test_repro_sha256_target_documented_and_matches(tmp_path):
    # The lockfile must state which file the hash is for; model.safetensors is the target.
    import yaml
    repro = yaml.safe_load((REPO/"REPRO.yaml").read_text())
    assert repro["weights"]["sha256"], "REPRO.yaml missing weights sha256"
    # weights present locally? skip the bytes check if not (CI has no weights)
    w = Path("/root/models/OvisOCR2/model.safetensors")
    if w.exists():
        import hashlib
        assert hashlib.sha256(w.read_bytes()).hexdigest() == repro["weights"]["sha256"]

def test_check_env_cpu_only_help():
    r = subprocess.run([sys.executable, str(REPO/"scripts"/"check-environment.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--cpu-only" in r.stdout
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/test_repro_inputs.py -q` → FAIL.

- [ ] **Step 3: Implement `scripts/check-environment.py`**:

```python
#!/usr/bin/env python3
"""Verify the active Python can serve OvisOCR2 (or, with --cpu-only, that the
repo+engine are importable without a GPU). Clear errors; never silent."""
from __future__ import annotations
import argparse, sys

def _ok(m): print(f"  [OK] {m}")
def _fail(m): print(f"  [FAIL] {m}"); return 1

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify the OvisOCR2-ROCm environment")
    p.add_argument("--cpu-only", action="store_true", help="skip GPU/vLLM checks (CI)")
    p.add_argument("--weights", default="/root/models/OvisOCR2")
    a = p.parse_args(argv)
    rc = 0
    import platform; _ok(f"Python {platform.python_version()}")
    try:
        import torch; _ok(f"torch {torch.__version__}")
    except Exception as e:
        rc = _fail(f"torch import: {e}")
    if not a.cpu_only:
        try:
            import vllm; _ok(f"vLLM {vllm.__version__}")
            from vllm.model_executor.models.registry import ModelRegistry
            reg = "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs()
            (_ok if reg else _fail)("Qwen3_5ForConditionalGeneration registered") ; rc |= 0 if reg else 1
            if hasattr(torch.version, "hip") and torch.version.hip:
                _ok(f"HIP/ROCm {torch.version.hip}")
            else:
                rc = _fail("no HIP/ROCm in this torch build")
            if not torch.cuda.is_available():
                rc = _fail("torch.cuda not available (no GPU visible)")
            else:
                _ok(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            rc = _fail(f"vLLM check: {e}")
    from pathlib import Path
    w = Path(a.weights)
    if w.exists():
        _ok(f"weights at {w}") if (w/"model.safetensors").exists() else (rc := _fail(f"weights missing model.safetensors at {w}"))
    elif not a.cpu_only:
        print(f"  [INFO] no weights at {w} (set --weights)")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement `scripts/verify-reproduction-inputs.py`**:

```python
#!/usr/bin/env python3
"""Verify REPRO.yaml reproduction inputs: weights sha256 (target: model.safetensors),
dataset revision, engine version, git commit. Exits 1 on mismatch."""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify REPRO.yaml reproduction inputs")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--weights", default="/root/models/OvisOCR2")
    a = p.parse_args(argv); rc = 0
    repro = yaml.safe_load((Path(a.repo)/"REPRO.yaml").read_text())
    print(f"[verify] model={repro['model_id']} overall={repro['overall']} engine_image={repro['environment']['image']}")
    wdir = Path(a.weights)
    target = wdir / "model.safetensors"
    if target.exists():
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h == repro["weights"]["sha256"]:
            print(f"  [OK] sha256(model.safetensors) == REPRO ({h[:12]}…)")
        else:
            print(f"  [FAIL] sha256 mismatch: got {h[:12]}… want {repro['weights']['sha256'][:12]}…"); rc = 1
    else:
        print(f"  [INFO] {target} not present locally — cannot verify weights hash")
    print(f"  weights revision: {repro['weights']['revision']}")
    print(f"  dataset revision: {repro['dataset']['revision']}  gt_sha256: {repro['dataset']['gt_sha256']}")
    print(f"  expected engine: {repro['environment']['image']}")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Implement `scripts/bootstrap-linux.sh`**:

```bash
#!/usr/bin/env bash
# Idempotent OvisOCR2-ROCm Linux provisioning. Non-destructive. Pinned installer tag.
set -euo pipefail
VENV="${VENV:-/root/venvs/vllm-0221b}"
WEIGHTS="${OVISOCR2_WEIGHTS:-/root/models/OvisOCR2}"
INSTALLER_TAG="${INSTALLER_TAG:-v1.0.0}"
INSTALLER_DIR="${INSTALLER_DIR:-/root/src/rocm-vllm-installer}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "[bootstrap] VENV=$VENV  WEIGHTS=$WEIGHTS  INSTALLER_TAG=$INSTALLER_TAG"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[bootstrap] building qwen3_5-capable vLLM venv at $VENV"
  if [ ! -d "$INSTALLER_DIR/.git" ]; then
    git clone https://github.com/AIwork4me/rocm-vllm-installer.git "$INSTALLER_DIR"
  fi
  ( cd "$INSTALLER_DIR" && git checkout "$INSTALLER_TAG" && VENV="$VENV" VLLM_VERSION=v0.22.1 bash install.sh )
else
  echo "[bootstrap] venv exists: $VENV"
fi

"$VENV/bin/python" "$REPO/scripts/check-environment.py" --weights "$WEIGHTS"

if [ ! -f "$WEIGHTS/model.safetensors" ]; then
  echo "[bootstrap] pinning weights revision -> $WEIGHTS"
  "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("ATH-MaaS/OvisOCR2", revision="65c619d374b55d4152e85150fc1b003700bc1f0c", local_dir="$WEIGHTS")
PY
fi

"$VENV/bin/python" "$REPO/scripts/verify-reproduction-inputs.py" --weights "$WEIGHTS" || true
echo "[bootstrap] done. Use $VENV/bin/python to run the adapter."
```

- [ ] **Step 6: Rewrite `adapter/setup/00-install-deps.sh`** (unify venv name, clone-not-curl, `VENV=` override):

```bash
#!/usr/bin/env bash
# OvisOCR2-ROCm — Linux/ROCm provisioning (thin wrapper over scripts/bootstrap-linux.sh).
set -euo pipefail
VENV="${VENV:-/root/venvs/vllm-0221b}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
echo "[00-install-deps] delegating to scripts/bootstrap-linux.sh (VENV=$VENV)"
exec bash "$REPO/scripts/bootstrap-linux.sh"
```

- [ ] **Step 7: Run tests to verify pass** — `pytest tests/test_repro_inputs.py -q` → PASS. `bash -n scripts/bootstrap-linux.sh && bash -n adapter/setup/00-install-deps.sh` (syntax OK).

- [ ] **Step 8: Checkpoint (no commit)** — `python scripts/check-environment.py --cpu-only` (expect OK for python/torch, INFO for weights), `python scripts/verify-reproduction-inputs.py` (expect OK sha256 if weights present). Suggested message: `feat(scripts): check-environment, bootstrap-linux (pinned clone, vllm-0221b), verify-reproduction-inputs; unify VENV default`.

---

## Task 6: `pyproject.toml` + CI (P0-5)

**Files:**
- Modify: `pyproject.toml`, `.github/workflows/ci.yml`

- [ ] **Step 1: Verify engine CPU-installability before pinning** — `Run: pip install --dry-run "omnidocbench-rocm==0.3.2" 2>&1 | head`. If it resolves without requiring ROCm torch, proceed with `==0.3.2`; if not, fall back to `==0.3.2` with a note and a `--no-deps` CI install path. Record the outcome in the commit-suggest message.

- [ ] **Step 2: Rewrite `pyproject.toml`**:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ovisocr2-rocm"
version = "1.0"
description = "OvisOCR2 (0.8B) end-to-end document parser on AMD ROCm via vLLM — an omnidocbench-rocm adapter."
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.11"
authors = [{ name = "AIwork4me" }]
classifiers = [
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = ["omnidocbench-rocm==0.3.2", "python-dotenv>=1.0"]

[project.urls]
Repository = "https://github.com/AIwork4me/OvisOCR2-ROCm"

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6", "build>=1.2", "pyyaml>=6"]

[tool.hatch.build.targets.wheel]
packages = ["adapter"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 3: Rewrite `.github/workflows/ci.yml`**:

```yaml
name: CI
on: [push, pull_request]
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest -q
      - run: python -m build
      - run: omnidocbench-rocm conformance .
      - run: python scripts/check-environment.py --cpu-only
      - name: Smoke demo
        run: bash examples/run_demo.sh
      - name: README path sanity
        run: |
          python - <<'PY'
          from pathlib import Path
          for f in ["examples/demo.png","adapter/run_adapter.py","eval/configs/omnidocbench_v16.yaml","REPRO.yaml"]:
              assert (Path(".")/f).exists(), f"README references missing path: {f}"
          print("README paths OK")
          PY
```

- [ ] **Step 4: Verify locally (CPU)** — `pip install -e ".[dev]"` then `ruff check . && pytest -q && python -m build && omnidocbench-rocm conformance .` → all pass. (CI badge is added in Task 8 only after this workflow would run — do not claim it has run.)

- [ ] **Step 5: Checkpoint (no commit)** — Suggested message: `ci: real gate (ruff+pytest+build+conformance) on py3.11/3.12; pyproject build-system/metadata/pin==0.3.2`.

---

## Task 7: Makefile developer experience

**Files:** Modify: `Makefile`

- [ ] **Step 1: Rewrite `Makefile`** (preserve existing targets' intent; add explicit ones):

```makefile
PLATFORM ?= linux-rocm
VERSION  ?= v16
REVISION ?= 2b161d0
MODEL_ID ?= ovisocr2
VENV     ?= /root/venvs/vllm-0221b
# eval defaults to the REAL backend; smoke is opt-in via demo-smoke.
BACKEND  ?= vllm
CDM ?= 1
RESUME ?= 0
CDM_FLAG = $(if $(filter 1,$(CDM)),--cdm,)
RESUME_FLAG = $(if $(filter 1,$(RESUME)),--skip-existing,)

PY = $(VENV)/bin/python

install-dev:
	pip install -e ".[dev]"

setup-linux:
	VENV=$(VENV) bash adapter/setup/00-install-deps.sh

check:
	ruff check . && pytest -q && python -m build && omnidocbench-rocm conformance .

smoke-test:
	python -m pytest

demo-smoke:
	omnidocbench-rocm infer --adapter adapter/run_adapter.py --img-dir examples --out-dir $$(mktemp -d) --platform $(PLATFORM) --backend smoke

demo-real:
	HIP_VISIBLE_DEVICES=0 $(PY) adapter/run_adapter.py --img-dir examples --out-dir /tmp/ovisocr2-demo --platform linux-rocm --backend vllm --limit-pages 1

eval-linux:
	omnidocbench-rocm run --stage all --platform linux-rocm --version $(VERSION) --revision $(REVISION) \
	  --adapter adapter/run_adapter.py --model-id $(MODEL_ID) --backend $(BACKEND) \
	  --git-commit $$(git rev-parse HEAD) --results-dir results/omnidocbench/$(VERSION)/linux-rocm \
	  $(CDM_FLAG) $(RESUME_FLAG)

eval-windows:
	@echo "windows-hip real inference is unsupported (community-wanted: no Qwen3-Next GDN HIP-SDK path)."; exit 1

conformance:
	omnidocbench-rocm conformance . && echo CONFORMANT

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache adapter/__pycache__ tests/__pycache__
	find . -name '*.pyc' -delete

publish: conformance
```

- [ ] **Step 2: Verify** — `make check && make smoke-test && make demo-smoke && make eval-windows` (the last must exit 1 with the unsupported message). `make eval-linux -n` (dry-run) shows `--backend vllm`.

- [ ] **Step 3: Checkpoint (no commit)** — Suggested message: `build(makefile): install-dev/check/demo-smoke/demo-real/eval-linux(vllm)/conformance/build/clean; eval-windows fails explicitly`.

---

## Task 8: Docs sync — P0-7 leftovers, P0-1, ADR-2, P0-6, REPRO

**Files:** `README.md`, `README.zh-CN.md`, `reproduce.md`, `REPRO.yaml`, `model_card.json`, new `analysis/formula-gap/*`.

- [ ] **Step 1: Fix the wrong attribution leftover** — `README.md:14-15`: replace `"the small CDM gap vs the paper's 96.58 is a verified vLLM-version artifact, see docs/known-gaps.md"` with `"the small CDM gap vs the paper's 96.58 is a model-inherent, version-independent formula-segmentation difference (verified by a 0.19.0-vs-0.22.1 A/B: CDM 95.41 on both); see docs/known-gaps.md"`. Apply the equivalent fix to `README.zh-CN.md:5` (公式 CDM 差距是模型固有的、与版本无关的公式切分差异; 经 0.19.0 与 0.22.1 A/B 验证, 两者 CDM 均为 95.41).

- [ ] **Step 2: 95.88 primary + 95.87 noted (ADR-2)** — in `model_card.json.note`, after "Reproduced Overall 95.88", append: `" (vLLM 0.22.1, the card's pinned version; the earlier 0.19.0 run scored 95.87 — CDM identical at 95.41 on both, confirming the gap is model-inherent)."` Leave `overall: 95.88` and `official_reference.delta_pp: -0.70` as-is.

- [ ] **Step 3: Install path (P0-1)** — in `README.md`/`README.zh-CN.md` Install sections, replace the `bash <(curl …)` line with:
```bash
# 1. ROCm vLLM 0.22.1 venv (one-time; ~1-2 h build). Clone the installer (it needs its patches/ dir):
git clone --branch v1.0.0 https://github.com/AIwork4me/rocm-vllm-installer.git
cd rocm-vllm-installer && VENV=/root/venvs/vllm-0221b VLLM_VERSION=v0.22.1 bash install.sh
# Verify qwen3_5 is registered:
/root/venvs/vllm-0221b/bin/python -c "from vllm.model_executor.models.registry import ModelRegistry as m; print('Qwen3_5ForConditionalGeneration' in m.get_supported_archs())"  # -> True
```
Set `VENV=/root/venvs/vllm-0221b` consistently in README, README.zh-CN, reproduce.md quickstart.

- [ ] **Step 4: Throughput honesty (P0-6)** — in `README.md`/`README.zh-CN.md` Known-Gaps and `reproduce.md`, change the throughput sentence to: `"Full-set inference is observed at ≈ 1 h on one W7900 (≈ 30 min sharded across two) — a manual measurement, not CI-derived; per-page latency is not recorded in the published bundle (see docs/known-gaps.md)."`

- [ ] **Step 5: Real sharding example (P0-4)** — in `reproduce.md`, replace the racy `for s in 0 1; do … --skip-existing & done` block with:
```bash
# Deterministic 2-GPU sharding (each shard writes its own subdir; merge validates coverage):
for s in 0 1; do HIP_VISIBLE_DEVICES=$s "$VENV/bin/python" adapter/run_adapter.py \
  --img-dir "$DATASET/images" --out-dir predictions/ovisocr2 \
  --platform linux-rocm --backend vllm --num-shards 2 --shard-index $s & done; wait
python scripts/merge-shards.py --input-root predictions/ovisocr2 \
  --out-dir predictions/ovisocr2-merged --expected-images "$DATASET/images"
# Score from the merged dir.
```
Add a one-line note: the old `--skip-existing` concurrent loop was **not** sharding (race on the same out-dir) and is removed.

- [ ] **Step 6: REPRO.yaml git_commit** — change `REPRO.yaml:31` `git_commit: "139959c..."` → `git_commit: "fdef86674d1519301ff9c8a29133750f173180a9"` (the real run commit). Leave overall (95.88), weights sha256, image (vllm-0221b) as-is.

- [ ] **Step 7: `analysis/formula-gap/` skeleton (P0-7)** — create:
  - `.gitkeep`
  - `README.md`: documents the A/B methodology — to attribute the CDM gap, hold these fixed: (1) same weights revision, (2) same images, (3) same prompt, (4) same pixel preprocessing (448²–2880²), (5) same batch size, (6) same scoring commit (`2b161d0`), (7) only vLLM version / GDN backend varies, (8) save per-formula CDM, (9) only claim version-attribution if 0.22.1 recovers CDM. State the current finding: 0.19.0 vs 0.22.1 → CDM 95.41 on both → **model-inherent, version-independent**. **Do not fabricate any 0.22.1 result file** beyond what is already committed.
  - `compare.py`: a stub that loads two per-formula CDM JSON files and prints the delta per formula + the count where delta > 0 (with a `if __name__=="__main__"` argparse). No fabricated inputs.
  - `sample_manifest.example.json`: `{"weights_revision": "65c619d...", "prompt_hash": "<sha256 of adapter prompt>", "versions": {"0.19.0": null, "0.22.1": null}, "note": "fill per_formula_cdm paths when re-running"}`.

- [ ] **Step 8: Conformance re-check** — `omnidocbench-rocm conformance .` → CONFORMANT. Verify both READMEs still contain the literal words `Install`, `Demo`, `Evaluation`, `Reproducibility`, `Known Gaps`. `grep -RniE "verified vLLM-version artifact|bash <\(curl" . --exclude-dir=.git` → no hits.

- [ ] **Step 9: Checkpoint (no commit)** — Suggested message: `docs: correct CDM attribution to model-inherent; vllm-0221b install via pinned clone; 95.88 primary + 95.87 noted; observed throughput; real sharding example; REPRO git_commit; analysis/formula-gap skeleton`.

---

## Task 9: Final verification (CPU bar) + manual GPU sanity run

- [ ] **Step 1: Full CPU bar** — run, record exit codes:
```bash
cd /workspace/OvisOCR2-ROCm
ruff check .                       # expect: clean
pytest -q                          # expect: all green (count the tests)
python -m build                    # expect: wheel + sdist built
omnidocbench-rocm conformance .    # expect: CONFORMANT
make smoke-test                    # expect: green
bash examples/run_demo.sh          # expect: smoke output OK
python adapter/run_adapter.py --help
python scripts/merge-shards.py --help
python scripts/check-environment.py --help
python scripts/verify-reproduction-inputs.py --help
python -c "from omnidocbench_rocm.bundle_validator import recompute_overall; import json; print(recompute_overall(json.load(open('results/omnidocbench/v16/linux-rocm/ovisocr2_v16_quick_match_cdm_metric_result.json'))))"  # expect: 95.88
```

- [ ] **Step 2: Manual GPU sanity (Layer 2, honest limits)** — pick a free partition (`rocm-smi --showmeminfo vram`; use an idle one, e.g. GPU 1 or 3), then:
```bash
HIP_VISIBLE_DEVICES=<idle> /root/venvs/vllm-0221b/bin/python adapter/run_adapter.py \
  --img-dir <a few images, e.g. examples + copies> --out-dir /tmp/gpu-sanity \
  --platform linux-rocm --backend vllm --batch-size 4
cat /tmp/gpu-sanity/_performance.json   # confirm real inference_wall_seconds, gpu, max_memory_allocated_mb
```
Expect: real Markdown output, `_performance.json` with non-null `inference_wall_seconds` / `gpu` / `max_memory_allocated_mb`. **Honest limits to state in the report:** 0 dataset images here → cannot reproduce 95.88 / full scoring; shared box → no reliable throughput; if no partition is free or it OOMs, say so — do not fake.

- [ ] **Step 3: Acceptance behaviors** — confirm (each via the tests above): (1) default CLI is unambiguous (eval→vllm, demo-smoke→smoke); (2) smoke imports no vllm; (3) `.env.local` honored, shell wins; (4) bad backend fails; (5) windows-hip+vllm fails; (6) two shards disjoint+complete; (7) merge catches a missing page; (8) a bad image doesn't abort the run; (9) skipped pages excluded from throughput; (10) CI runs pytest (not just an import).

- [ ] **Step 4: Checkpoint (no commit)** — Produce the final report per the spec's required output format (audit / changes / behavior deltas / test evidence with counts / unverified items / remaining work / risks / quality score).

---

## Self-Review (run after writing)

**Spec coverage:** P0-1 → Task 5 (+docs Task 8); P0-2 → Task 1 + Task 3 (CLI); P0-3 → Task 3; P0-4 → Task 2 (primitives) + Task 3 (adapter) + Task 4 (merge) + Task 8 (doc); P0-5 → Task 6; P0-6 → Task 3 (timing/sidecars) + Task 8 (doc); P0-7 leftovers → Task 8. Makefile/pyproject → Tasks 6–7. All 4 ADR decisions encoded. ✅
**Placeholders:** none — every code step shows real code; the `_run_batch` placeholder was corrected in Step 3b.
**Type consistency:** `Config` fields (Task 1) match `resolve()` usage (Task 1) and `run_adapter` reads (`cfg.weights_dir`, `cfg.batch_size`, `cfg.num_shards`, `cfg.shard_index`, `cfg.skip_existing`, `cfg.temperature`, `cfg.limit_pages`); `_run_batch` returns `(image, text, seconds, error)` 4-tuples consumed verbatim in Task 3c; `merge_shards(input_root, expected_images, out_dir)` signature matches Task 4 CLI. ✅
