"""CLI tests, focused on exit codes because that is the CI contract."""

from __future__ import annotations

import json

import pytest

from exlex.cli import EXIT_OK, EXIT_POLICY, EXIT_USAGE, main


def test_no_args_prints_help(capsys):
    assert main([]) == EXIT_USAGE
    assert "usage" in capsys.readouterr().out.lower()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "exlex" in capsys.readouterr().out


def test_doctor(capsys):
    assert main(["doctor"]) == EXIT_OK
    assert "exlex session" in capsys.readouterr().out


def test_doctor_json(capsys):
    assert main(["doctor", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["achieved"]


def test_levels_lists_residual_risk(capsys):
    assert main(["levels"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "still can" in out
    assert "signing key" in out


def test_levels_json(capsys):
    assert main(["levels", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert str(Level_value_attested()) in payload


def Level_value_attested():
    from exlex.levels import Level

    return Level.ATTESTED.value


def test_harden_probe(capsys):
    code = main(["harden"])
    assert code in (EXIT_OK, EXIT_POLICY)
    assert "platform" in capsys.readouterr().out


def test_harden_no_fail_forces_zero(capsys):
    assert main(["harden", "--no-fail"]) == EXIT_OK
    capsys.readouterr()


def test_harden_json(capsys):
    main(["harden", "--json", "--no-fail"])
    assert "checks" in json.loads(capsys.readouterr().out)


def test_attest_reports_failure_without_raising(capsys):
    code = main(["attest"])
    assert code in (EXIT_OK, EXIT_POLICY)
    out = capsys.readouterr().out
    assert "attestation:" in out


def test_attest_json(capsys):
    main(["attest", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "ok" in payload and "residual" in payload


def test_audit_clean_file(tmp_path, capsys):
    path = tmp_path / "clean.py"
    path.write_text("def f(t):\n    return t.to('cuda:0')\n")
    assert main(["audit", str(path), "--quiet-blind-spots"]) == EXIT_OK
    assert "no findings" in capsys.readouterr().out


def test_audit_dirty_file_fails(tmp_path, capsys):
    path = tmp_path / "leaky.py"
    path.write_text(
        "import exlex\n\n@exlex.secret\ndef f(t):\n    return t.cpu()\n"
    )
    assert main(["audit", str(path), "--quiet-blind-spots"]) == EXIT_POLICY
    assert "transfer" in capsys.readouterr().out


def test_audit_prints_blind_spots_by_default(tmp_path, capsys):
    path = tmp_path / "clean.py"
    path.write_text("x = 1\n")
    main(["audit", str(path)])
    assert "blind spots" in capsys.readouterr().out


def test_audit_json(tmp_path, capsys):
    path = tmp_path / "leaky.py"
    path.write_text("def f(t):\n    return t.item()\n")
    main(["audit", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["rule"] == "transfer"


def test_audit_fail_on_any(tmp_path):
    path = tmp_path / "medium.py"
    path.write_text("def f(t):\n    return t.numpy()\n")
    assert main(["audit", str(path), "--fail-on", "any", "--quiet-blind-spots"]) == EXIT_POLICY


def test_audit_fail_on_never(tmp_path):
    path = tmp_path / "leaky.py"
    path.write_text("def f(t):\n    return t.cpu()\n")
    assert main(["audit", str(path), "--fail-on", "never", "--quiet-blind-spots"]) == EXIT_OK


def test_audit_missing_path_is_usage_error(capsys):
    assert main(["audit", "/nope/does/not/exist"]) == EXIT_USAGE


def test_guard_reports_state(capsys):
    code = main(["guard"])
    assert code in (EXIT_OK, EXIT_POLICY)
    assert "residency guard" in capsys.readouterr().out


def test_protect_prints_every_concern(capsys):
    assert main(["protect"]) == EXIT_OK
    out = capsys.readouterr().out
    for concern in ("scripts", "data", "correctness", "operations", "metadata"):
        assert concern in out


def test_protect_json_is_machine_readable(capsys):
    assert main(["protect", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["remote"] is True
    assert "metadata" in payload["claims"] or "metadata" in payload["unaddressed"]


def test_protect_local_reports_no_remote(capsys):
    assert main(["protect", "--local"]) == EXIT_OK
    payload_line = "local" in capsys.readouterr().out
    assert payload_line


def test_protect_fail_on_open_flags_unprotected(capsys):
    # Metadata is never protected, so this must be a policy failure.
    assert main(["protect", "--fail-on-open"]) == EXIT_POLICY
    capsys.readouterr()


def test_protect_always_names_the_standing_residuals(capsys):
    main(["protect"])
    out = capsys.readouterr().out
    assert "no configuration removes" in out
    assert "circuit structure" in out
