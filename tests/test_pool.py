"""Tests for hardware detection and pool computation."""

from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local.config import ConfigError
from snakemake_executor_plugin_shared_local.pool import (
    Pool,
    apply_reserve,
    compute_pool,
    detect_cpu_count,
    detect_mem_total_mib,
)


def test_detect_cpu_count_positive():
    assert detect_cpu_count() >= 1


def test_detect_mem_total_from_fixture(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       527824324 kB\nMemFree:  1234 kB\n")
    # 527824324 kB -> MiB
    assert detect_mem_total_mib(meminfo) == 527824324 // 1024


def test_detect_mem_total_missing_key(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("Something: 1 kB\n")
    with pytest.raises(ConfigError, match="MemTotal"):
        detect_mem_total_mib(meminfo)


def test_detect_mem_total_unreadable(tmp_path: Path):
    with pytest.raises(ConfigError, match="Cannot read"):
        detect_mem_total_mib(tmp_path / "nope")


@pytest.mark.parametrize(
    "total,expected", [(100, 90), (10, 9), (1, 1), (1000, 900), (3, 3)]
)
def test_apply_reserve(total, expected):
    assert apply_reserve(total, 0.10) == expected


def test_apply_reserve_never_below_one():
    # A tiny machine must still yield a usable pool of at least one core.
    assert apply_reserve(1, 0.9) == 1


def test_apply_reserve_rejects_bad_fraction():
    with pytest.raises(ConfigError, match="reserve_fraction"):
        apply_reserve(100, 1.0)
    with pytest.raises(ConfigError, match="reserve_fraction"):
        apply_reserve(100, -0.1)


def test_apply_reserve_rejects_nonpositive_total():
    with pytest.raises(ConfigError):
        apply_reserve(0)


def test_compute_pool_uses_explicit_values():
    pool = compute_pool(pool_cores=32, pool_mem_mib=4096)
    assert pool.cores == 32
    assert pool.mem_mib == 4096


def test_compute_pool_detects_when_unspecified():
    pool = compute_pool()
    assert pool.cores >= 1
    # Memory detection should succeed on Linux; if not, mem_mib may be None.
    assert pool.mem_mib is None or pool.mem_mib >= 1


def test_compute_pool_rejects_invalid_explicit_values():
    with pytest.raises(ConfigError, match="pool_cores"):
        compute_pool(pool_cores=0)
    with pytest.raises(ConfigError, match="pool_mem"):
        compute_pool(pool_mem_mib=0)


def test_pool_as_dict_includes_only_known_memory():
    assert Pool(cores=4, mem_mib=None).as_dict() == {"cores": 4}
    assert Pool(cores=4, mem_mib=100).as_dict() == {"cores": 4, "mem_mib": 100}


def test_compute_pool_survives_unreadable_meminfo(monkeypatch):
    """CPU coordination must still work when memory cannot be detected."""
    import snakemake_executor_plugin_shared_local.pool as pool_module

    def boom(*args, **kwargs):
        raise ConfigError("no meminfo")

    monkeypatch.setattr(pool_module, "detect_mem_total_mib", boom)
    pool = pool_module.compute_pool()
    assert pool.mem_mib is None
    assert pool.cores >= 1
