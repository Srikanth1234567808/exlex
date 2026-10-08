"""Command line interface.

Subcommands map onto the questions a user actually has:

* ``exlex doctor``     what is true on this host right now
* ``exlex levels``     what each level means, including what it does not mean
* ``exlex harden``     probe, and optionally fix, host configuration
* ``exlex attest``     run attestation and show the verdict
* ``exlex audit``      static analysis over a path
* ``exlex guard``      check that a process is running under the guard
* ``exlex protect``    what is protected per concern, for a remote workload

Exit codes are chosen for CI use: 0 clean, 1 policy violation, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from . import audit as _audit
from . import attest as _attest
from . import harden as _harden
from .concerns import REPORT_ORDER
from .errors import ExlexError
from .levels import SPECS
from .session import doctor as _doctor

EXIT_OK = 0
EXIT_POLICY = 1
EXIT_USAGE = 2


def _cmd_doctor(args: argparse.Namespace) -> int:
    print(_doctor(as_json=args.json))
    return EXIT_OK


def _cmd_levels(args: argparse.Namespace) -> int:
    if args.json:
        print(
            json.dumps(
                {
                    lvl.value: {
                        "name": s.name,
                        "prevents": sorted(s.prevents),
                        "residual": list(s.residual),
                        "requires": list(s.requires),
                    }
                    for lvl, s in SPECS.items()
                },
                indent=2,
            )
        )
        return EXIT_OK
    for level in sorted(SPECS, key=lambda item: item.value):
        s = SPECS[level]
        print(f"\n{s.name}  (level {level.value})")
        print("  stops an attacker from:")
        for item in sorted(s.prevents):
            print(f"    + {item}")
        print("  a determined attacker still can:")
        for item in s.residual:
            print(f"    - {item}")
        if s.requires:
            print("  requires:")
            for item in s.requires:
                print(f"    * {item}")
    print()
    return EXIT_OK


def _cmd_harden(args: argparse.Namespace) -> int:
    report = _harden.probe()
    if args.apply:
        report = _harden.apply(report, dry_run=False)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("after apply:")
            for check in report.checks:
                print("  " + str(check))
        return EXIT_OK if report.ok else EXIT_POLICY
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"platform: {report.platform}")
        for check in report.checks:
            print("  " + str(check))
        if not report.ok:
            print("\nfix with: sudo exlex harden --apply")
    return EXIT_OK if report.ok or args.no_fail else EXIT_POLICY


def _cmd_attest(args: argparse.Namespace) -> int:
    result = _attest.detect()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        state = "VERIFIED" if result.ok else "FAILED"
        print(f"attestation: {state}  (backend: {result.backend})")
        for name, ok in result.checks.items():
            print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
        if result.measurements:
            print("  measurements:")
            for key, value in result.measurements.items():
                print(f"    {key} = {value}")
        if result.reasons:
            print("  reasons:")
            for reason in result.reasons:
                print(f"    - {reason}")
        if result.ok:
            print("  residual risk even when verified:")
            for item in result.residual:
                print(f"    - {item}")
    return EXIT_OK if result.ok else EXIT_POLICY


def _cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.exists():
        print(f"exlex: no such path: {root}", file=sys.stderr)
        return EXIT_USAGE
    findings = _audit.audit_path(root) if root.is_dir() else _audit.audit_file(root)
    if args.json:
        print(json.dumps(_audit.to_dicts(findings), indent=2))
    else:
        for finding in findings:
            print(f"{root.name}:{finding}")
        if not findings:
            print("no findings")
        if not args.quiet_blind_spots:
            print("\nknown blind spots of this analysis:")
            for spot in _audit.KNOWN_BLIND_SPOTS:
                print(f"  - {spot}")
    high = [f for f in findings if f.severity == "high"]
    if args.fail_on == "high" and high:
        return EXIT_POLICY
    if args.fail_on == "any" and findings:
        return EXIT_POLICY
    return EXIT_OK


def _cmd_protect(args: argparse.Namespace) -> int:
    """Report what is protected for a remote workload, per concern."""
    from .protect import protect, protect_local

    if args.local:
        report = protect_local()
    else:
        report = protect()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.explain())
        if not args.json:
            print()
            print("residuals that no configuration removes:")
            for item in report.residuals():
                print(f"  - {item}")
    # Unprotected is a finding, not a clean bill of health: a caller gating on
    # the exit code wants to know when a concern is open.
    open_concerns = [c for c in REPORT_ORDER if not report.is_protected(c)]
    if args.fail_on_open and open_concerns:
        return EXIT_POLICY
    return EXIT_OK


def _cmd_guard(args: argparse.Namespace) -> int:
    from .residency import _GUARD  # noqa: PLC0415

    state = "installed" if _GUARD.installed else "not installed"
    print(f"residency guard: {state}")
    print(json.dumps(_GUARD.summary(), indent=2))
    if _GUARD.violations:
        print(f"{len(_GUARD.violations)} violation(s) recorded this process")
        return EXIT_POLICY
    return EXIT_OK if _GUARD.installed else EXIT_POLICY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exlex",
        description="Defence-in-depth controls for ML workloads on untrusted GPU hosts.",
    )
    parser.add_argument("--version", action="version", version=f"exlex {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("doctor", help="report what is true on this host now")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("levels", help="explain each assurance level and its limits")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_levels)

    p = sub.add_parser("harden", help="probe, and optionally fix, host configuration")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--apply", action="store_true", help="write sysctls (needs root)")
    p.add_argument(
        "--no-fail",
        action="store_true",
        help="always exit 0, for informational use in a pipeline",
    )
    p.set_defaults(func=_cmd_harden)

    p = sub.add_parser("attest", help="run attestation and show the verdict")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_attest)

    p = sub.add_parser("audit", help="static analysis for host-boundary crossings")
    p.add_argument("path", help="file or directory to audit")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--fail-on",
        choices=("high", "any", "never"),
        default="high",
        help="exit non-zero on this severity or above (default: high)",
    )
    p.add_argument(
        "--quiet-blind-spots",
        action="store_true",
        help="omit the list of documented analysis limitations",
    )
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("guard", help="report the residency guard state")
    p.set_defaults(func=_cmd_guard)

    p = sub.add_parser(
        "protect",
        help="report what is protected per concern, and what still leaks",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--local",
        action="store_true",
        help="assume the workload runs on your own machine, with no remote host",
    )
    p.add_argument(
        "--fail-on-open",
        action="store_true",
        help="exit non-zero when any concern is not fully protected",
    )
    p.set_defaults(func=_cmd_protect)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.func(args)
    except ExlexError as exc:
        print(f"exlex: {exc}", file=sys.stderr)
        return EXIT_POLICY
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
