"""Tests for NUMA topology detection, node selection and binding.

Binding is a placement feature, not a limit. These tests therefore check that a
node is chosen sensibly and that impossible requests degrade to unbound execution
instead of failing the workflow.
"""

import os

import pytest

from snakemake_executor_plugin_shared_local import numa
from snakemake_executor_plugin_shared_local.numa import (
    NumaNode,
    NumaTopology,
    detect_topology,
    reset_current_thread,
    select_cpus,
    select_node,
)


def node(index, ncpus=8, total=1000, free=500):
    return NumaNode(
        index=index,
        cpus=list(range(index * ncpus, (index + 1) * ncpus)),
        mem_total_mib=total,
        mem_free_mib=free,
    )


class TestTopologyDetection:
    def test_detects_real_topology(self):
        topo = detect_topology()
        # The development machine has two nodes; CI containers may have one.
        assert len(topo.nodes) >= 1
        for entry in topo.nodes:
            assert entry.cpus, "every detected node must list CPUs"
            assert entry.mem_total_mib > 0

    def test_availability_requires_more_than_one_node(self):
        assert not NumaTopology(nodes=[node(0)]).available
        assert NumaTopology(nodes=[node(0), node(1)]).available

    def test_empty_topology_is_unavailable(self):
        topo = NumaTopology(nodes=[])
        assert not topo.available
        assert topo.total_cpus() == 0
        assert select_node(topo, cores=1, mem_mib=None) is None

    def test_missing_directory_yields_empty_topology(self, tmp_path):
        assert detect_topology(tmp_path / "absent").nodes == []

    def test_reads_synthetic_sysfs(self, tmp_path):
        node0 = tmp_path / "node0"
        node1 = tmp_path / "node1"
        for entry, cpus, total in ((node0, "0-3", 1000), (node1, "4,6", 2000)):
            entry.mkdir()
            (entry / "cpulist").write_text(cpus)
            (entry / "meminfo").write_text(
                f"Node 0 MemTotal:       {total * 1024} kB\n"
                f"Node 0 MemFree:        {total * 512} kB\n"
            )
        topo = detect_topology(tmp_path)
        assert [n.index for n in topo.nodes] == [0, 1]
        assert topo.nodes[0].cpus == [0, 1, 2, 3]
        assert topo.nodes[1].cpus == [4, 6]
        assert topo.nodes[0].mem_total_mib == 1000
        assert topo.nodes[0].mem_free_mib == 500

    def test_ignores_non_node_entries(self, tmp_path):
        (tmp_path / "node0").mkdir()
        (tmp_path / "node0" / "cpulist").write_text("0-1")
        (tmp_path / "node0" / "meminfo").write_text("Node 0 MemTotal: 1024 kB\n")
        (tmp_path / "possible").write_text("0-1")
        topo = detect_topology(tmp_path)
        assert len(topo.nodes) == 1

    def test_skips_nodes_without_cpus(self, tmp_path):
        (tmp_path / "node0").mkdir()
        (tmp_path / "node0" / "meminfo").write_text("Node 0 MemTotal: 1024 kB\n")
        assert detect_topology(tmp_path).nodes == []


class TestNodeSelection:
    def test_picks_the_node_with_most_free_memory(self):
        topo = NumaTopology(nodes=[node(0, free=100), node(1, free=900)])
        assert select_node(topo, cores=1, mem_mib=50).index == 1

    def test_respects_requested_core_count(self):
        small = node(0, ncpus=2, free=900)
        big = node(1, ncpus=16, free=100)
        topo = NumaTopology(nodes=[small, big])
        # 8 cores do not fit on the 2-CPU node regardless of free memory.
        assert select_node(topo, cores=8, mem_mib=None).index == 1

    def test_respects_memory_capacity(self):
        topo = NumaTopology(
            nodes=[node(0, total=100, free=90), node(1, total=5000, free=10)]
        )
        assert select_node(topo, cores=1, mem_mib=1000).index == 1

    def test_returns_none_when_no_node_can_host_the_job(self):
        """Oversized jobs must degrade to unbound rather than fail."""
        topo = NumaTopology(nodes=[node(0, total=1000), node(1, total=1000)])
        assert select_node(topo, cores=1, mem_mib=5000) is None

    def test_returns_none_when_cores_exceed_every_node(self):
        topo = NumaTopology(nodes=[node(0, ncpus=4), node(1, ncpus=4)])
        assert select_node(topo, cores=16, mem_mib=None) is None

    def test_accounts_for_already_used_memory(self):
        topo = NumaTopology(nodes=[node(0, free=1000), node(1, free=600)])
        # Node 0 is busier once in-flight placements are considered.
        chosen = select_node(topo, cores=1, mem_mib=None, used_per_node={0: 900})
        assert chosen.index == 1

    def test_memory_constraint_is_optional(self):
        topo = NumaTopology(nodes=[node(0), node(1)])
        assert select_node(topo, cores=1, mem_mib=None) is not None


class TestBinding:
    def test_binding_restricts_cpu_affinity(self):
        topo = detect_topology()
        if not topo.available:
            pytest.skip("single-node machine has nothing to bind")
        target = topo.nodes[0]
        original = set(os.sched_getaffinity(0))
        numa.capture_original_affinity()
        try:
            result = numa.bind_current_thread(target, cores=2)
            if not result.applied:
                pytest.skip("binding unavailable in this environment")
            assert os.sched_getaffinity(0) <= set(target.cpus)
        finally:
            os.sched_setaffinity(0, original)

    def test_binding_uses_at_most_requested_cpus(self):
        topo = detect_topology()
        if not topo.available:
            pytest.skip("single-node machine has nothing to bind")
        original = set(os.sched_getaffinity(0))
        numa.capture_original_affinity()
        try:
            result = numa.bind_current_thread(topo.nodes[0], cores=2)
            if not result.applied:
                pytest.skip("binding unavailable")
            assert len(os.sched_getaffinity(0)) <= 2
        finally:
            os.sched_setaffinity(0, original)

    def test_binding_propagates_to_child_processes(self):
        """Snakemake spawns job children, so binding must be inherited."""
        topo = detect_topology()
        if not topo.available:
            pytest.skip("single-node machine has nothing to bind")
        import subprocess

        original = set(os.sched_getaffinity(0))
        numa.capture_original_affinity()
        try:
            result = numa.bind_current_thread(topo.nodes[0], cores=2)
            if not result.applied:
                pytest.skip("binding unavailable")
            out = subprocess.run(
                ["sh", "-c", "grep Cpus_allowed_list /proc/self/status"],
                capture_output=True,
                text=True,
            ).stdout
            allowed = out.split(":")[1].strip()
            assert allowed != "0-127" or len(topo.nodes[0].cpus) == 128
        finally:
            os.sched_setaffinity(0, original)

    def test_reset_restores_original_affinity(self):
        topo = detect_topology()
        if not topo.available:
            pytest.skip("single-node machine has nothing to bind")
        original = set(os.sched_getaffinity(0))
        numa.capture_original_affinity()
        try:
            result = numa.bind_current_thread(topo.nodes[0], cores=1)
            if not result.applied:
                pytest.skip("binding unavailable")
            reset_current_thread()
            assert os.sched_getaffinity(0) == original
        finally:
            os.sched_setaffinity(0, original)

    def test_reset_is_safe_without_prior_binding(self):
        numa.reset_current_thread()  # must not raise


class TestSyscallWrapper:
    def test_bind_memory_rejects_unavailable_libc(self, monkeypatch):
        binder = numa._NumaBinder()
        monkeypatch.setattr(binder, "_load", lambda: None)
        assert binder.bind_memory([0]) is False

    def test_bind_cpus_rejects_empty_list(self):
        assert numa._NumaBinder().bind_cpus([]) is False

    def test_bind_memory_reset_path(self, monkeypatch):
        calls = []

        class FakeLibc:
            def syscall(self, *args):
                calls.append(args)
                return 0

        binder = numa._NumaBinder()
        monkeypatch.setattr(binder, "_load", lambda: FakeLibc())
        assert binder.bind_memory([0]) is True
        assert binder.bind_memory([]) is True
        assert len(calls) == 2
        # The reset must use MPOL_DEFAULT, not MPOL_BIND.
        assert calls[1][1] == numa.MPOL_DEFAULT


class TestReadCpuList:
    def test_parses_ranges_and_singles(self, tmp_path):
        path = tmp_path / "cpulist"
        path.write_text("0-2,5,7-8\n")
        assert numa._read_cpu_list(path) == [0, 1, 2, 5, 7, 8]

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "cpulist"
        path.write_text("")
        assert numa._read_cpu_list(path) is None

    def test_malformed_returns_none(self, tmp_path):
        path = tmp_path / "cpulist"
        path.write_text("a-b")
        assert numa._read_cpu_list(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert numa._read_cpu_list(tmp_path / "nope") is None


class TestCpuSelection:
    """Picking *which* CPUs of a node to bind to.

    Binding every job of an instance to the node's lowest-numbered CPUs stacks all
    of them onto one core: a real 128-CPU run had 74 concurrent 1-core jobs, all
    with ``Cpus_allowed_list: 0``, and the host sat at 12.8% utilisation.
    """

    def test_single_core_jobs_spread_across_the_node(self):
        n = node(0, ncpus=8)  # cpus 0..7
        chosen = [select_cpus(n, cores=1, offset=i) for i in range(8)]
        assert len({tuple(c) for c in chosen}) == 8, (
            f"1-core jobs must not share a CPU: {chosen}"
        )
        assert sorted(cpu for c in chosen for cpu in c) == list(range(8))

    def test_selection_stays_inside_the_node(self):
        n = node(1, ncpus=8)
        for offset in range(20):
            chosen = select_cpus(n, cores=2, offset=offset)
            assert len(chosen) == 2
            assert set(chosen) <= set(n.cpus)

    def test_wraps_around_without_leaving_the_node(self):
        n = node(0, ncpus=4)
        for offset in range(10):
            assert set(select_cpus(n, cores=1, offset=offset)) <= set(n.cpus)

    def test_selection_respects_requested_core_count(self):
        n = node(0, ncpus=16)
        for cores in (1, 2, 4, 8):
            assert len(select_cpus(n, cores=cores, offset=0)) == cores

    def test_oversized_request_falls_back_to_the_whole_node(self):
        """A job wider than the node cannot be spread; use everything available."""
        n = node(0, ncpus=4)
        assert sorted(select_cpus(n, cores=64, offset=0)) == sorted(n.cpus)

    def test_offset_zero_is_not_special_cased_to_one_cpu(self):
        """The first job of a node must not own a CPU the next job will also use."""
        n = node(0, ncpus=4)
        first = select_cpus(n, cores=1, offset=0)
        second = select_cpus(n, cores=1, offset=1)
        assert first != second, (
            "consecutive 1-core jobs must get different CPUs, otherwise a whole "
            "instance funnels onto one core"
        )

    def test_zero_or_negative_cores_still_yields_one_cpu(self):
        n = node(0, ncpus=4)
        assert len(select_cpus(n, cores=0, offset=0)) == 1
        assert len(select_cpus(n, cores=-3, offset=0)) == 1

    def test_empty_node_returns_nothing(self):
        empty = NumaNode(index=0, cpus=[], mem_total_mib=1, mem_free_mib=1)
        assert select_cpus(empty, cores=1, offset=0) == []


class TestBindingSpreadsJobs:
    def test_two_single_core_bindings_do_not_use_the_same_cpu(self):
        topo = detect_topology()
        if not topo.available:
            pytest.skip("single-node machine has nothing to bind")
        target = topo.nodes[0]
        original = set(os.sched_getaffinity(0))
        numa.capture_original_affinity()
        try:
            first = numa.bind_current_thread(target, cores=1, offset=0)
            if not first.applied:
                pytest.skip("binding unavailable in this environment")
            cpus_a = set(os.sched_getaffinity(0))

            second = numa.bind_current_thread(target, cores=1, offset=1)
            if not second.applied:
                pytest.skip("binding unavailable in this environment")
            cpus_b = set(os.sched_getaffinity(0))
        finally:
            os.sched_setaffinity(0, original)

        assert cpus_a != cpus_b, (
            "two jobs on the same node were bound to the same CPU, which is the "
            "bug that capped a 128-core host at 13% utilisation"
        )
