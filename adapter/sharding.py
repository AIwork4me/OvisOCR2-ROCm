"""Backward-compat re-export — sharding lives in ``ovisocr2_rocm.sharding``."""
from __future__ import annotations

from ovisocr2_rocm.sharding import (  # noqa: F401
    IMG_EXT,
    MergeReport,
    merge_shards,
    select_shard,
    shard_dir,
    validate_shard_args,
)
