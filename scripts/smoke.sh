#!/usr/bin/env bash
# End-to-end CLI check. Not part of the test suite; a smoke test for a
# human or a release job. Prints the exit code of each subcommand, which is
# the part that matters for CI integration.
set -u
cd "$(dirname "$0")/.."
EXLEX=.venv/bin/exlex

run() {
  local label="$1"; shift
  "$@" >/dev/null 2>&1
  printf '%-34s exit=%s\n' "$label" "$?"
}

echo "--- exlex levels (first lines) ---"
$EXLEX levels 2>&1 | head -12
echo
echo "--- exit codes ---"
run "levels"              $EXLEX levels
run "doctor"              $EXLEX doctor
run "harden"              $EXLEX harden
run "harden --no-fail"    $EXLEX harden --no-fail
run "attest"              $EXLEX attest
run "guard"               $EXLEX guard
run "audit clean"         $EXLEX audit examples/quickstart.py --quiet-blind-spots
run "audit missing path"  $EXLEX audit /nope/nothing/here

printf 'import exlex\n\n@exlex.secret\ndef f(t):\n    return t.cpu()\n' >/tmp/exlex_smoke_leak.py
run "audit dirty (expect 1)" $EXLEX audit /tmp/exlex_smoke_leak.py --quiet-blind-spots

echo
echo "--- doctor json ---"
$EXLEX doctor --json 2>/dev/null | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print("achieved:", d["achieved"]); print("controls:", d["controls"]); print("residuals:", len(d["residual"]))'
rm -f /tmp/exlex_smoke_leak.py
