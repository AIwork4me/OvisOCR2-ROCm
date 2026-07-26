"""Deterministic sharding tests (P0-4): disjoint/cover, determinism, arg validation."""
from pathlib import Path

import pytest

from adapter import sharding as S


def imgs(n):
    return [Path(f"page_{i:03d}.png") for i in range(n)]


def test_two_shards_disjoint_and_cover():
    a = S.select_shard(imgs(10), 2, 0)
    b = S.select_shard(imgs(10), 2, 1)
    assert set(a) | set(b) == set(imgs(10))
    assert not (set(a) & set(b))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_n_shards_cover_all(n):
    parts = [S.select_shard(imgs(10), n, i) for i in range(n)]
    flat = [p for part in parts for p in part]
    assert len(flat) == 10 and set(flat) == set(imgs(10))


def test_deterministic_and_sorted():
    assert S.select_shard(imgs(10), 3, 1) == sorted(imgs(10))[1::3]


def test_shard_dir_layout():
    assert S.shard_dir(Path("/o"), 1, 0) == Path("/o")
    assert S.shard_dir(Path("/o"), 2, 0).name == "shard-00000-of-00002"
    assert S.shard_dir(Path("/o"), 2, 1).name == "shard-00001-of-00002"


def test_bad_args_rejected():
    with pytest.raises(ValueError):
        S.select_shard(imgs(5), 0, 0)
    with pytest.raises(ValueError):
        S.select_shard(imgs(5), 2, 2)
