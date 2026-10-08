"""Hardening probe and apply tests."""

from __future__ import annotations

import pytest

from exlex import harden
from exlex.errors import HardeningError


def test_probe_returns_report_with_named_controls():
    report = harden.probe()
    names = {c.name for c in report.checks}
    assert "sysctl:vm.swappiness" in names
    assert "swap:disabled" in names
    assert "ulimit:memlock" in names
    assert "coredump:enabled" in names


def test_check_string_shows_observed_and_expected():
    report = harden.probe()
    check = report.checks[0]
    text = str(check)
    assert "observed" in text and "want" in text


def test_report_serialises():
    payload = harden.probe().to_dict()
    assert set(payload) == {"platform", "ok", "checks"}
    assert isinstance(payload["checks"], list)


def test_core_pattern_piped_is_accepted():
    assert harden._core_pattern_ok("|/bin/false") is True
    assert harden._core_pattern_ok("|/usr/share/apport/apport") is True


def test_core_pattern_to_disk_is_rejected():
    assert harden._core_pattern_ok("/var/crash/core") is False
    assert harden._core_pattern_ok("core") is False
    assert harden._core_pattern_ok(None) is False


def test_apply_dry_run_changes_nothing(monkeypatch):
    report = harden.probe()
    before = [(c.name, c.observed) for c in report.checks]
    harden.apply(report, dry_run=True)
    after = [(c.name, c.observed) for c in harden.probe().checks]
    assert before == after


def test_apply_non_dry_run_returns_report(monkeypatch):
    monkeypatch.setattr(harden, "probe", lambda **kw: harden.HardeningReport())
    result = harden.apply(dry_run=False)
    assert isinstance(result, harden.HardeningReport)


def test_require_raises_listing_failures(monkeypatch):
    report = harden.probe()
    for check in report.checks:
        check.passed = False
    with pytest.raises(HardeningError, match="hardening incomplete"):
        harden.require(report)


def test_require_passes_when_all_ok(monkeypatch):
    report = harden.probe()
    for check in report.checks:
        check.passed = True
    assert harden.require(report).ok


def test_run_helper_never_raises():
    rc, _out, _err = harden._run(["definitely-not-a-real-binary-xyz"])
    assert rc == 127


def test_gpu_inventory_degrades_gracefully():
    devices = harden.gpu_inventory()
    assert isinstance(devices, list) and devices


def test_console_telemetry_degrades_gracefully():
    info = harden.console_telemetry()
    assert "note" in info or info.get("available") is False


def test_is_large():
    assert harden._is_large(str(128 * 1024 * 1024)) is True
    assert harden._is_large("1024") is False
    assert harden._is_large("not-a-number") is False


def test_rlimit_returns_none_or_string():
    value = harden._rlimit("RLIMIT_MEMLOCK")
    assert value is None or isinstance(value, str)


def test_rlimit_unknown_constant_is_none():
    assert harden._rlimit("RLIMIT_NOT_A_REAL_LIMIT") is None
