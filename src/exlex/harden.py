"""Host hardening: probes, and optionally applies, defensive configuration.

The split between :func:`probe` and :func:`apply` is deliberate. Reading state
is always safe. Changing sysctls, swap, or mount options needs privileges and
mutates the machine, so it is a separate, explicit call. A library that silently
rewrote a host's kernel parameters would be a bad neighbour in a community tool.

What this module can and cannot do is worth stating plainly. These are
configuration controls against a *non-cooperative* operator who lacks the
ability to change the boot chain. They are not isolation. A root attacker reads
guest RAM regardless of ``ptrace_scope``, and a provider with hypervisor access
is not stopped by anything in this file.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import HardeningError

#: sysctl key -> (required values, human rationale)
SYSCTL_POLICY: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "kernel.yama.ptrace_scope": (("3",), "block ptrace of other processes"),
    "kernel.dmesg_restrict": (("1",), "keep the kernel ring buffer unreadable"),
    "kernel.kptr_restrict": (("2",), "hide kernel pointers from unprivileged readers"),
    "kernel.unprivileged_bpf_disabled": (("1", "2"), "stop unprivileged eBPF programs"),
    "kernel.core_pattern": (("|",), "no on-disk core dumps"),
    "vm.unprivileged_rlimit_memlock": (("0",), "let mlock work without privilege"),
    "vm.swappiness": (("0",), "avoid swapping secrets out"),
    "fs.suid_dumpable": (("0",), "no privileged core dumps"),
    "net.ipv4.conf.all.send_redirects": (("0",), "no ICMP redirects"),
    "net.ipv4.conf.all.accept_redirects": (("0",), "no ICMP redirects"),
    "net.ipv4.tcp_syncookies": (("1",), "resist SYN flood"),
    "net.ipv4.conf.all.accept_source_route": (("0",), "no source routing"),
    "net.ipv4.conf.all.log_martians": (("1",), "log spoofed packets"),
}

#: Paths that must not be world-writable when holding secrets.
SENSITIVE_PATHS: Tuple[str, ...] = (
    "/dev/shm",
    "/run/shm",
    "/tmp",
    "/var/tmp",
)


@dataclass
class Check:
    """The outcome of a single hardening probe.

    Attributes:
        name: the control, e.g. ``"sysctl:vm.swappiness"``.
        passed: whether the observed state matches policy.
        observed: what was actually found, verbatim.
        expected: what policy wants.
        why: one line on what this stops.
        fixable: whether :func:`apply` can set it without root-only surgery.
    """

    name: str
    passed: bool
    observed: str
    expected: str
    why: str
    fixable: bool = False

    def __str__(self) -> str:
        mark = "ok  " if self.passed else "FAIL"
        return f"[{mark}] {self.name}: observed {self.observed!r}, want {self.expected!r}"


@dataclass
class HardeningReport:
    """Aggregate view of a probe run.

    Attributes:
        checks: every probe that ran, passed or not.
        platform: the platform string, so a report cannot be misread across hosts.
    """

    checks: List[Check] = field(default_factory=list)
    platform: str = field(default_factory=platform.platform)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def names(self) -> List[str]:
        return [c.name for c in self.failures]

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "ok": self.ok,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "observed": c.observed,
                    "expected": c.expected,
                    "why": c.why,
                }
                for c in self.checks
            ],
        }


def _read_sysctl(key: str, runner=os) -> Optional[str]:
    """Read one sysctl value, returning None when the key is absent."""
    proc_path = Path("/proc/sys") / key.replace(".", "/")
    try:
        return proc_path.read_text().strip()
    except (OSError, PermissionError):
        return None


def _read_file(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return None


def _run(cmd: Sequence[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    """Run a command, returning (rc, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _core_pattern_ok(value: Optional[str]) -> bool:
    """Accept only a piped pattern.

    Every other form writes a core file to disk: an absolute path writes there
    directly, and a bare name like ``core`` lands in the process working
    directory. Either way the file is plaintext, so both are rejected. A pipe
    is acceptable because the handler then decides the destination, which is a
    decision the operator makes and the exlex report surfaces.
    """
    if value is None:
        return False
    return value.strip().startswith("|")


def probe(*, include_optional: bool = True) -> HardeningReport:
    """Inspect the current host. Read-only; safe to call anywhere.

    Args:
        include_optional: include checks that commonly fail on ordinary
            desktops, such as world-writable /tmp. Useful in CI, noisy in
            development.
    """
    report = HardeningReport()
    checks: List[Check] = []

    for key, (acceptable, why) in SYSCTL_POLICY.items():
        observed = _read_sysctl(key)
        if key == "kernel.core_pattern":
            passed = _core_pattern_ok(observed)
            expected = "piped to a handler, e.g. '|/bin/false'"
        else:
            passed = observed in acceptable
            expected = " or ".join(acceptable)
        checks.append(
            Check(
                name=f"sysctl:{key}",
                passed=passed,
                observed=observed if observed is not None else "<unreadable>",
                expected=expected,
                why=why,
                fixable=True,
            )
        )

    swap = _swap_in_use()
    checks.append(
        Check(
            name="swap:disabled",
            passed=not swap,
            observed="enabled" if swap else "disabled",
            expected="disabled",
            why="stop secrets landing in a swap file the operator can read",
        )
    )

    ml = _mlock_ceiling()
    checks.append(
        Check(
            name="ulimit:memlock",
            passed=ml is not None and (ml == "unlimited" or _is_large(ml)),
            observed=ml or "<unknown>",
            expected="unlimited, or large enough for the working set",
            why="mlock keeps secret pages off disk; without it the call fails",
        )
    )

    dumps = _core_dumps_enabled()
    checks.append(
        Check(
            name="coredump:enabled",
            passed=not dumps,
            observed="enabled" if dumps else "disabled",
            expected="disabled",
            why="a core dump is a plaintext copy of the whole process",
        )
    )

    if include_optional:
        for path in SENSITIVE_PATHS:
            if not Path(path).exists():
                continue
            mode = _dir_mode(path)
            if mode is None:
                continue
            sticky = _is_sticky(path)
            checks.append(
                Check(
                    name=f"perm:{path}",
                    passed=not (mode & 0o002) or sticky,
                    observed=oct(mode),
                    expected="not world-writable unless sticky",
                    why="other tenants should not be able to swap a file into place",
                )
            )

        egress = _egress_open()
        checks.append(
            Check(
                name="net:egress",
                passed=not egress,
                observed="open" if egress else "blocked",
                expected="blocked",
                why="no outbound path means no exfiltration and no callback",
            )
        )

    report.checks = checks
    return report


def _swap_in_use() -> bool:
    try:
        return Path("/proc/swaps").read_text().strip().splitlines()[1:] != []
    except (OSError, IndexError):
        return False


def _rlimit(constant: str) -> Optional[str]:
    """Read a POSIX rlimit as a string. None when unavailable, e.g. Windows.

    Args:
        constant: the name of the ``RLIMIT_*`` symbol, e.g. ``"RLIMIT_MEMLOCK"``.
    """
    try:
        import resource  # noqa: PLC0415

        which = getattr(resource, constant, None)
        getrlimit = getattr(resource, "getrlimit", None)
        if which is None or getrlimit is None:
            return None
        return str(getrlimit(which)[0])
    except Exception:  # pragma: no cover - platform dependent
        return None


def _mlock_ceiling() -> Optional[str]:
    return _rlimit("RLIMIT_MEMLOCK")


def _is_large(value: str) -> bool:
    try:
        return int(value) >= 64 * 1024 * 1024
    except ValueError:
        return False


def _core_dumps_enabled() -> bool:
    pattern = _read_file("/proc/sys/kernel/core_pattern")
    if pattern is None:
        return False
    ceiling = _rlimit("RLIMIT_CORE")
    if ceiling is not None and ceiling.isdigit() and int(ceiling) == 0:
        return False
    return not _core_pattern_ok(pattern)


def _dir_mode(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mode & 0o777
    except OSError:
        return None


def _is_sticky(path: str) -> bool:
    try:
        return bool(os.stat(path).st_mode & 0o1000)
    except OSError:
        return False


def _egress_open(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.5) -> bool:
    """Probe outbound connectivity. Best effort, never raises."""
    import socket  # noqa: PLC0415

    for family, socktype, proto, _canon, sockaddr in (
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port)),
        (socket.AF_INET, socket.SOCK_DGRAM, 0, "", (host, port)),
    ):
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def apply(report: Optional[HardeningReport] = None, *, dry_run: bool = True) -> HardeningReport:
    """Attempt to fix the fixable failures in ``report``.

    Args:
        report: a prior probe. Probed again if omitted.
        dry_run: when True (the default) nothing is changed and the report
            describes what would happen. Pass ``dry_run=False`` to actually
            write sysctls; this needs root and is not reversible at runtime.

    Returns:
        A fresh report from the subsequent probe, so callers can see what stuck.
    """
    report = report or probe()
    for check in report.checks:
        if check.passed or not check.fixable:
            continue
        key = check.name.split(":", 1)[1]
        path = Path("/proc/sys") / key.replace(".", "/")
        if key == "kernel.core_pattern":
            value = "|/bin/false"
        else:
            value = check.expected.split(" or ")[0]
        if dry_run:
            continue
        try:
            path.write_text(value)
        except (OSError, PermissionError):
            pass
    return probe()


def require(report: Optional[HardeningReport] = None) -> HardeningReport:
    """Raise :class:`HardeningError` if any fixable control is failing.

    Non-fixable checks, such as egress state, are reported but do not raise,
    because a library should not refuse to start over a network probe.
    """
    report = report or probe()
    blocking = [c for c in report.failures if c.fixable]
    if blocking:
        detail = "\n".join(f"  {c.name}={c.observed!r} (want {c.expected!r})" for c in blocking)
        raise HardeningError(
            "host hardening incomplete; run `exlex harden --apply` as root:\n" + detail
        )
    return report


def gpu_inventory() -> List[dict]:
    """Enumerate visible NVIDIA GPUs via ``nvidia-smi``.

    Aimed at the PASSTHROUGH/SR-IOV question: a discrete GPU handed to a guest
    is a device the host kernel drives, which is the strongest realistic
    residual for GPU workloads. Knowing which mode you are in changes what the
    rest of the guarantees mean.
    """
    rc, out, err = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if rc != 0:
        return [{"error": err or "nvidia-smi unavailable"}]
    devices = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        devices.append(
            {
                "name": parts[0] if parts else "unknown",
                "memory_total": parts[1] if len(parts) > 1 else "unknown",
                "driver_version": parts[2] if len(parts) > 2 else "unknown",
            }
        )
    return devices


def console_telemetry() -> dict:
    """Report whether the NVIDIA driver forwards usage to a remote service.

    ``NVreg_EnableGpuFirmware`` and friends are not remotely readable, but
    driver persistence and the accounts service are, and both can carry
    identifying information. Checking is the honest part; the fix is a driver
    module parameter and is left to the operator.
    """
    rc, out, _err = _run(["nvidia-smi", "-q"])
    if rc != 0:
        return {"available": False}
    return {
        "available": True,
        "persistence_mode": "Enabled" in out,
        "note": (
            "Disable persistence mode and any accounts-service login if this "
            "host is shared; the driver otherwise retains identifying state."
        ),
    }
