"""Session, level, and public API tests."""

from __future__ import annotations

import json

import pytest

import exlex
from exlex.errors import HardeningError, PolicyError, ResidencyViolation
from exlex.levels import Level, spec, weakest
from exlex.session import Session, SessionReport, _coerce_level

from conftest import FakeTensor, make_fake_torch


def _guard():
    """A guard bound to the fake torch, so no real torch is needed."""
    return exlex.ResidencyGuard(torch_module=make_fake_torch())


# ------------------------------------------------------------------- levels

def test_levels_are_ordered():
    assert Level.NONE < Level.HARDENED < Level.RESIDENT < Level.ATTESTED


def test_weakest_returns_lowest():
    assert weakest([Level.ATTESTED, Level.HARDENED, Level.NONE]) is Level.NONE
    assert weakest([Level.RESIDENT, Level.ATTESTED]) is Level.RESIDENT
    assert weakest([]) is Level.NONE


def test_every_level_documents_residual_risk():
    for level in Level:
        s = spec(level)
        if level is Level.NONE:
            continue
        assert s.residual, f"{s.name} claims no residual risk, which is never true"
        assert s.requires or level is Level.NONE


def test_attested_spec_names_the_real_trust_anchor():
    joined = " ".join(spec(Level.ATTESTED).residual).lower()
    assert "signing key" in joined
    assert "vram" in joined
    assert "physical" in joined


def test_coerce_level_accepts_names_and_ints():
    assert _coerce_level("resident") is Level.RESIDENT
    assert _coerce_level("ATTESTED") is Level.ATTESTED
    assert _coerce_level(2) is Level.RESIDENT
    assert _coerce_level(Level.HARDENED) is Level.HARDENED
    with pytest.raises(PolicyError):
        _coerce_level("nonsense")


# ------------------------------------------------------------------ session

def test_disabling_a_lower_level_caps_the_achieved_level():
    """The ladder is cumulative, so skipping a rung caps the result at NONE.

    Residency depends on hardening (mlock, no swap, no coredumps). Claiming
    RESIDENT with hardening disabled would be a false claim, so the session
    reports NONE.
    """
    sess = Session(level="attested", harden=False, attest=False, guard=_guard())
    report = sess.start()
    assert report.requested is Level.ATTESTED
    assert report.achieved is Level.NONE
    assert "hardening was not requested" in (report.downgrade_reason or "")


def test_residency_only_claim_requires_hardening_to_pass(monkeypatch):
    from exlex import harden as harden_module

    clean = harden_module.probe()
    for check in clean.checks:
        check.passed = True
    monkeypatch.setattr(harden_module, "probe", lambda **kw: clean)
    sess = Session(level=Level.RESIDENT, attest=False, guard=_guard())
    report = sess.start()
    # The guard is installed, but the process-wide default is still a fresh
    # unpatched torch, so assert on the control rather than on device state.
    assert report.controls["residency"] is True
    assert report.controls["hardening"] is True
    assert report.achieved >= Level.HARDENED


def test_failed_hardening_prevents_resident_claim(monkeypatch):
    """A failed rung is not a rung. Nothing above it may be claimed."""
    from exlex import harden as harden_module

    dirty = harden_module.probe()
    for check in dirty.checks:
        check.passed = False
    monkeypatch.setattr(harden_module, "probe", lambda **kw: dirty)
    sess = Session(level=Level.RESIDENT, attest=False, guard=_guard())
    report = sess.start()
    assert report.achieved is Level.NONE
    assert report.controls["hardening"] is False
    assert "cannot be claimed without passing hardening" in (report.downgrade_reason or "")


def test_requested_level_is_never_granted_without_attestation():
    sess = Session(level=Level.ATTESTED, guard=_guard())
    report = sess.start()
    assert report.achieved is not Level.ATTESTED
    assert report.controls["attestation"] is False


def test_none_level_reports_none():
    sess = Session(level=Level.NONE)
    report = sess.start()
    assert report.achieved is Level.NONE


def test_session_start_is_idempotent():
    sess = Session(level=Level.RESIDENT, harden=False, guard=_guard())
    first = sess.start()
    second = sess.start()
    assert first is second


def test_session_context_manager_stops_guard():
    guard = _guard()
    with exlex.session(level=Level.RESIDENT, harden=False, guard=guard):
        assert guard.installed
    assert not guard.installed


def test_on_missing_ignore_is_accepted():
    sess = Session(level=Level.HARDENED, on_missing="ignore", guard=_guard())
    assert sess.start().achieved is not None


def test_invalid_on_missing_rejected():
    with pytest.raises(ValueError):
        Session(on_missing="shrug")


def test_require_hardening_raises_when_incomplete(monkeypatch):
    from exlex import harden as harden_module

    report = harden_module.probe()
    for check in report.checks:
        check.passed = False
    monkeypatch.setattr(harden_module, "probe", lambda **kw: report)
    with pytest.raises(HardeningError):
        Session(level=Level.HARDENED, require_hardening=True).start()


def test_environ_is_populated():
    env = {}
    Session(level=Level.RESIDENT, harden=False, environ=env, guard=_guard()).start()
    assert env.get("WANDB_MODE") == "offline"
    assert env.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def test_environ_none_is_safe():
    Session(level=Level.RESIDENT, harden=False, environ=None, guard=_guard()).start()


# ------------------------------------------------------------ public API

def test_secret_decorator_blocks_transfer(fake_torch):
    guard = exlex.ResidencyGuard(torch_module=fake_torch)
    previous = exlex.set_guard(guard)
    guard.install()
    try:

        @exlex.secret
        def export(model):
            return model.cpu()

        with pytest.raises(ResidencyViolation):
            export(FakeTensor(1.0))
    finally:
        guard.uninstall()
        exlex.set_guard(previous)


def test_secret_decorator_with_parens(fake_torch):
    guard = exlex.ResidencyGuard(torch_module=fake_torch)
    previous = exlex.set_guard(guard)
    guard.install()
    try:

        @exlex.secret()
        def export(model):
            return model.numpy()

        with pytest.raises(ResidencyViolation):
            export(FakeTensor(1.0))
    finally:
        guard.uninstall()
        exlex.set_guard(previous)


def test_allow_requires_reason_at_api_level():
    with pytest.raises(ValueError):
        with exlex.allow(""):
            pass


def test_resident_context_opens_region(fake_torch):
    guard = exlex.ResidencyGuard(torch_module=fake_torch)
    previous = exlex.set_guard(guard)
    guard.install()
    try:
        tensor = FakeTensor(1.0)
        with exlex.resident():
            with pytest.raises(ResidencyViolation):
                tensor.cpu()
        tensor.item()
    finally:
        guard.uninstall()
        exlex.set_guard(previous)


def test_set_guard_returns_previous():
    original = exlex.get_guard()
    replacement = exlex.ResidencyGuard()
    assert exlex.set_guard(replacement) is original
    assert exlex.get_guard() is replacement
    exlex.set_guard(original)


def test_public_api_surface():
    for name in ("session", "secret", "allow", "resident", "doctor", "Level"):
        assert hasattr(exlex, name)
    assert "Level" in exlex.__all__


def test_version_present():
    assert exlex.__version__


# ----------------------------------------------------------------- reports

def test_report_explain_includes_residual_and_downgrade():
    report = SessionReport(
        requested=Level.ATTESTED,
        achieved=Level.RESIDENT,
        controls={"attestation": False},
        residual=["gpu vram is readable"],
        downgrade_reason="attestation unavailable",
    )
    text = report.explain()
    assert "requested=attested" in text
    assert "achieved=resident" in text
    assert "gpu vram" in text
    assert "attestation unavailable" in text


def test_report_to_json_is_valid():
    report = SessionReport(requested=Level.NONE, achieved=Level.NONE)
    assert json.loads(report.to_json())["requested"] == "none"


def test_doctor_returns_text():
    output = exlex.doctor()
    assert "exlex session" in output
    assert "GPUs" in output


def test_doctor_json_is_valid():
    assert json.loads(exlex.doctor(as_json=True))["achieved"] in (
        "none",
        "hardened",
        "resident",
        "attested",
    )
