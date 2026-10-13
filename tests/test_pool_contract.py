"""The pool/request contract: both sides must speak the same dimensions.

This file exists because of a real bug. ``Pool.as_dict()`` published the memory
dimension as ``mem_mib`` while ``normalize_resources()`` produced ``mem_mb``.
``policy.fits()`` treats a dimension absent from the pool as *unconstrained* (see
its docstring), so the mismatch did not raise - it silently disabled memory
arbitration altogether. A pool of 100 MiB admitted a job asking for 26000 MiB.

These tests pin the invariant that makes that class of bug impossible to
reintroduce quietly: every dimension the pool can publish must be a dimension a
job request can use, and the memory dimension must be counted on both sides.
"""

import pytest

from snakemake_executor_plugin_shared_local.lease import (
    AdmissionImpossible,
    LeaseManager,
    normalize_resources,
)
from snakemake_executor_plugin_shared_local.namespace import resolve_namespace
from snakemake_executor_plugin_shared_local.policy import fits
from snakemake_executor_plugin_shared_local.pool import (
    MEMORY_DIMENSION,
    Pool,
    canonicalize_pool,
)


def _namespace(tmp_path):
    return resolve_namespace("user", explicit_dir=str(tmp_path / "leases"))


def test_pool_and_request_agree_on_dimension_names():
    """No pool dimension may be a name a job request can never produce."""
    pool_dims = set(Pool(cores=4, mem_mb=100).as_dict())
    request_dims = set(normalize_resources({"mem_mb": 8}, threads=1))
    assert pool_dims & request_dims == {"cores", MEMORY_DIMENSION}


def test_memory_dimension_is_not_a_derived_suffix():
    """``normalize_resources`` drops ``*_mib`` aliases; the pool must not use one."""
    assert not MEMORY_DIMENSION.endswith("_mib")


def test_oversized_memory_request_is_refused(tmp_path):
    """The regression: a small memory pool must refuse a large memory job."""
    manager = LeaseManager(
        _namespace(tmp_path),
        Pool(cores=116, mem_mb=100).as_dict(),
        logger=None,
        poll_interval=0.01,
    )
    request = normalize_resources({"mem_mb": 26000}, threads=16)
    with pytest.raises(AdmissionImpossible) as excinfo:
        manager.acquire("job1", request, rule="align")
    message = str(excinfo.value)
    assert "mem_mb=26000" in message
    assert "--shared-local-pool-mem" in message


def test_memory_is_actually_enforced_by_fits():
    """Guard the policy layer directly, independent of the lease machinery."""
    pool = Pool(cores=116, mem_mb=100).as_dict()
    assert fits({"cores": 16, "mem_mb": 50}, {}, pool) is True
    assert fits({"cores": 16, "mem_mb": 26000}, {}, pool) is False


def test_canonicalize_pool_maps_legacy_mem_mib():
    """A pool.json written by an older plugin must still be understood."""
    assert canonicalize_pool({"cores": 4, "mem_mib": 8192}) == {
        "cores": 4,
        MEMORY_DIMENSION: 8192,
    }


def test_canonicalize_pool_prefers_the_canonical_spelling():
    assert canonicalize_pool({"mem_mib": 7, MEMORY_DIMENSION: 9}) == {
        MEMORY_DIMENSION: 9
    }
    assert canonicalize_pool({MEMORY_DIMENSION: 9, "mem_mib": 7}) == {
        MEMORY_DIMENSION: 9
    }


def test_canonicalize_pool_keeps_and_normalizes_valid_values():
    assert canonicalize_pool({"cores": 4, "gpu": 1}) == {"cores": 4, "gpu": 1}
    assert canonicalize_pool({"cores": "8"}) == {"cores": 8}


def test_canonicalize_pool_rejects_unusable_values():
    """Dropping a bad dimension would silently remove that ceiling entirely."""
    from snakemake_executor_plugin_shared_local.config import ConfigError

    for bad in ({"cores": 0}, {"cores": -1}, {"gpu": "abc"}, {"x": True}, {"g": None}):
        with pytest.raises(ConfigError):
            canonicalize_pool(bad)


def test_legacy_pool_declaration_is_not_a_conflict(tmp_path):
    """Upgrading the plugin must not turn an existing namespace into an error."""
    import json

    namespace = _namespace(tmp_path)
    namespace.path.mkdir(parents=True, exist_ok=True)
    (namespace.path / "pool.json").write_text(
        json.dumps(
            {
                "version": 1,
                "host": "somehost",
                "pool": {"cores": 4, "mem_mib": 8192},
                "created_by": 0,
                "created_at": 0,
            }
        )
    )
    manager = LeaseManager(
        namespace,
        Pool(cores=4, mem_mb=8192).as_dict(),
        logger=None,
        poll_interval=0.01,
    )
    # Must not raise LeaseError: same pool, different spelling.
    manager.negotiate_pool()
    # ...and the canonical dimension is what constrains admission.
    assert manager.pool == {"cores": 4, MEMORY_DIMENSION: 8192}
