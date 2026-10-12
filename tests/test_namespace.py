"""Tests for namespace resolution and its permission handling.

Permission details are correctness-critical here, not cosmetics: a lease file that
is unreadable to peers makes their usage invisible and silently breaks the global
limit, and a directory missing the group/other bits means a shared scope never
coordinates at all.
"""

import os
import stat
from pathlib import Path

import pytest

from snakemake_executor_plugin_shared_local.config import ConfigError
from snakemake_executor_plugin_shared_local.namespace import (
    DIR_MODES,
    Namespace,
    resolve_namespace,
)


class TestExplicitDirectory:
    def test_uses_given_directory(self, tmp_path):
        target = tmp_path / "custom"
        target.mkdir()
        ns = resolve_namespace("user", explicit_dir=str(target))
        assert ns.path == target.absolute()
        assert ns.mode == "user"

    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "nested" / "deep"
        ns = resolve_namespace("user", explicit_dir=str(target))
        assert ns.path.is_dir()

    def test_rejects_file_path(self, tmp_path):
        target = tmp_path / "afile"
        target.write_text("x")
        with pytest.raises(ConfigError, match="not a directory"):
            resolve_namespace("user", explicit_dir=str(target))

    def test_expands_user_and_variables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NS_TEST", str(tmp_path / "fromenv"))
        ns = resolve_namespace("user", explicit_dir="$NS_TEST")
        assert ns.path == (tmp_path / "fromenv").absolute()

    def test_explicit_directory_permissions_are_not_changed(self, tmp_path):
        """An explicit path is the user's decision; we must not rewrite its mode."""
        target = tmp_path / "keepmode"
        target.mkdir(mode=0o755)
        os.chmod(target, 0o755)
        resolve_namespace("user", explicit_dir=str(target))
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o755


class TestUmaskHandling:
    """umask must not silently disable a coordination scope."""

    @pytest.mark.parametrize("mode", ["user", "group", "system"])
    def test_directory_mode_survives_restrictive_umask(
        self, tmp_path, mode, monkeypatch
    ):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        old = os.umask(0o077)
        try:
            ns = resolve_namespace(mode, explicit_dir=None)
        finally:
            os.umask(old)
        actual = stat.S_IMODE(os.stat(ns.path).st_mode)
        assert actual == DIR_MODES[mode], (
            f"umask 077 left {mode} scope at {oct(actual)}, "
            f"expected {oct(DIR_MODES[mode])}"
        )

    def test_group_scope_is_writable_by_group(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        old = os.umask(0o022)
        try:
            ns = resolve_namespace("group", explicit_dir=None)
        finally:
            os.umask(old)
        assert stat.S_IMODE(os.stat(ns.path).st_mode) & 0o020

    def test_stale_mode_on_our_directory_is_repaired(self, tmp_path, monkeypatch):
        """A directory created earlier under a different umask must be fixed."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ns = resolve_namespace("system", explicit_dir=None)
        os.chmod(ns.path, 0o700)  # simulate the pre-existing wrong mode
        repaired = resolve_namespace("system", explicit_dir=None)
        assert stat.S_IMODE(os.stat(repaired.path).st_mode) == DIR_MODES["system"]


class TestNamespaceProperties:
    def test_user_scope_is_private(self, tmp_path):
        ns = resolve_namespace("user", explicit_dir=str(tmp_path))
        assert not ns.is_shared

    def test_group_and_system_scopes_are_shared(self, tmp_path):
        for mode in ("group", "system"):
            ns = resolve_namespace(mode, explicit_dir=str(tmp_path))
            assert ns.is_shared

    def test_non_sticky_directory_allows_collection(self, tmp_path):
        ns = resolve_namespace("system", explicit_dir=str(tmp_path))
        assert ns.may_remove_foreign_entries

    def test_sticky_directory_forbids_removing_foreign_entries(self, tmp_path):
        sticky = tmp_path / "sticky"
        sticky.mkdir()
        os.chmod(sticky, 0o1777)
        ns = resolve_namespace("system", explicit_dir=str(sticky))
        assert ns.sticky
        assert not ns.may_remove_foreign_entries

    def test_lease_path_is_inside_directory(self, tmp_path):
        ns = resolve_namespace("user", explicit_dir=str(tmp_path))
        assert ns.lease_path("x.lease").parent == ns.path


class TestInvalidMode:
    def test_unknown_mode_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="Invalid namespace"):
            resolve_namespace("galaxy", explicit_dir=str(tmp_path))


class TestFallbackCandidates:
    def test_falls_back_when_first_base_is_unusable(self, tmp_path, monkeypatch):
        """If /dev/shm is unavailable, a later candidate must be used."""
        import snakemake_executor_plugin_shared_local.namespace as ns_module

        good = tmp_path / "goodbase"
        good.mkdir()
        bad = tmp_path / "badbase"  # deliberately absent and uncreatable
        monkeypatch.setattr(
            ns_module, "CANDIDATE_BASES", (str(bad / "nope" / "deeper"), str(good))
        )
        # Make the first candidate fail by pointing it at a path whose parent is a file.
        blocker = bad / "nope"
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text("not a directory")
        ns = resolve_namespace("user")
        assert str(ns.path).startswith(str(good))

    def test_raises_when_no_candidate_works(self, tmp_path, monkeypatch):
        import snakemake_executor_plugin_shared_local.namespace as ns_module

        blocker = tmp_path / "blocker"
        blocker.write_text("file")
        monkeypatch.setattr(ns_module, "CANDIDATE_BASES", (str(blocker / "sub"),))
        with pytest.raises(ConfigError, match="Could not establish"):
            resolve_namespace("user")


class TestDescribeIdentity:
    def test_reports_uid_gid_and_user(self):
        from snakemake_executor_plugin_shared_local.namespace import describe_identity

        info = describe_identity()
        assert info["uid"] == os.getuid()
        assert info["gid"] == os.getgid()
        assert isinstance(info["user"], str)


def test_namespace_is_a_dataclass_with_expected_fields(tmp_path):
    ns = Namespace(
        mode="user",
        path=tmp_path,
        dir_mode=0o700,
        sticky=False,
        uid=1,
        gid=1,
    )
    assert ns.mode == "user"
    assert Path(ns.path) == tmp_path
