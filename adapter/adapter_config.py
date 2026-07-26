"""Adapter configuration for OvisOCR2-ROCm.

Priority: explicit CLI > environment (incl. ``adapter/setup/.env.local``) >
defaults. CLI flags that override config arrive as ``None`` when unset, so an
absent flag never clobbers an env value. Backend is a closed set {smoke, vllm};
``windows-hip + vllm`` is rejected with an explicit error (no silent fallback).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
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


def _bool(s: object) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")


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
    """JSON-serializable view of the resolved config, for run_stats/sidecar provenance."""
    return asdict(cfg)
