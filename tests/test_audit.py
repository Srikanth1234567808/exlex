"""Static analysis tests, including the documented blind spots."""

from __future__ import annotations

import textwrap

import pytest

from exlex.audit import (
    KNOWN_BLIND_SPOTS,
    Finding,
    audit_file,
    audit_path,
    audit_source,
    to_dicts,
)


def _rules(source: str):
    return {f.rule for f in audit_source(textwrap.dedent(source))}


def test_finds_cpu_transfer_in_secret_function():
    source = """
        import exlex

        @exlex.secret
        def export(model):
            return model.weights.cpu()
    """
    findings = audit_source(textwrap.dedent(source))
    assert [f.rule for f in findings] == ["transfer"]
    assert findings[0].in_secret is True
    assert findings[0].severity == "high"


def test_finds_transfer_in_unmarked_function_at_lower_severity():
    source = """
        def helper(t):
            return t.numpy()
    """
    findings = audit_source(textwrap.dedent(source))
    assert findings and findings[0].severity == "medium"
    assert findings[0].in_secret is False


def test_to_cpu_is_flagged_but_to_cuda_is_not():
    flagged = audit_source("def f(t):\n    return t.to('cpu')\n")
    clean = audit_source("def f(t):\n    return t.to('cuda:0')\n")
    assert flagged
    assert not clean


def test_allow_block_suppresses_findings():
    source = """
        import exlex

        @exlex.secret
        def publish(t):
            with exlex.allow("aggregate only"):
                return t.item()
    """
    assert _rules(source) == set()


def test_findings_after_allow_block_are_still_flagged():
    source = """
        import exlex

        @exlex.secret
        def publish(t):
            with exlex.allow("aggregate only"):
                pass
            return t.numpy()
    """
    assert _rules(source) == {"transfer"}


def test_detects_serialisation_of_secrets():
    source = """
        import exlex

        @exlex.secret
        def save(model):
            torch.save(model.state_dict(), "ckpt.pt")
    """
    assert "serialize" in _rules(source)


def test_detects_network_sink_in_secret_region():
    source = """
        import exlex

        @exlex.secret
        def upload(secret_payload):
            requests.post("https://example.invalid", data=secret_payload)
    """
    assert "network" in _rules(source)


def test_detects_file_write_in_secret_region():
    source = """
        import exlex

        @exlex.secret
        def dump(secret_token):
            with open("/tmp/out", "w") as fh:
                fh.write(secret_token)
    """
    assert "file" in _rules(source)


def test_secret_variable_names_are_tracked_across_statements():
    source = """
        import exlex

        @exlex.secret
        def send():
            api_key = load()
            requests.post("https://example.invalid", data=api_key)
    """
    findings = audit_source(textwrap.dedent(source))
    assert [f.rule for f in findings] == ["network"]


def test_secret_region_does_not_leak_to_sibling_function():
    source = """
        import exlex

        @exlex.secret
        def guarded(t):
            return t.numpy()

        def unguarded(t):
            return t.numpy()
    """
    findings = audit_source(textwrap.dedent(source))
    assert len(findings) == 2
    assert {f.in_secret for f in findings} == {True, False}


def test_async_function_secret_marker():
    source = """
        import exlex

        @exlex.secret
        async def run(t):
            return t.cpu()
    """
    assert _rules(source) == {"transfer"}


def test_multiline_call_not_evaded_by_newlines():
    source = """
        import exlex

        @exlex.secret
        def f(t):
            return t.to(
                "cpu"
            )
    """
    assert _rules(source) == {"transfer"}


def test_plain_function_raises_syntax_error():
    with pytest.raises(SyntaxError):
        audit_source("def broken(:\n")


def test_documented_blind_spot_dynamic_attribute_is_real():
    source = """
        import exlex

        @exlex.secret
        def sneaky(t):
            return getattr(t, "cp" + "u")()
    """
    assert _rules(source) == set()


def test_non_python_file_skipped(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("x = t.cpu()")
    assert audit_file(path) == []


def test_audit_path_skips_vendor_dirs(tmp_path):
    vendor = tmp_path / ".venv" / "lib"
    vendor.mkdir(parents=True)
    (vendor / "bad.py").write_text("def f(t):\n    return t.cpu()\n")
    (tmp_path / "good.py").write_text("def f(t):\n    return t.to('cuda')\n")
    assert audit_path(tmp_path) == []


def test_to_dicts_shape():
    findings = audit_source("def f(t):\n    return t.cpu()\n")
    payload = to_dicts(findings)
    assert set(payload[0]) == {
        "rule",
        "severity",
        "lineno",
        "col",
        "call",
        "message",
        "in_secret",
    }


def test_finding_str_includes_region_tag():
    finding = Finding(
        rule="transfer",
        severity="high",
        lineno=3,
        col=4,
        call="t.cpu()",
        message="copies to host",
        in_secret=True,
    )
    assert "in secret region" in str(finding)


def test_blind_spots_are_documented():
    assert len(KNOWN_BLIND_SPOTS) >= 5
    assert any("compiled" in spot for spot in KNOWN_BLIND_SPOTS)
