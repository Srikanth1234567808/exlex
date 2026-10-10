# Contributing to exlex

This is an open research project. Attack-first: if it leaks, prove it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest --cov=exlex --cov-report=term-missing --cov-fail-under=85
```

Needs TenSEAL for the FHE path: `pip install 'exlex[fhe]'` (Python 3.11+).
Without it those tests skip; core stays dependency-free.

## What to work on

See `ROADMAP.md` and issues labeled `good-first-issue` / `attack`.
Good contributions: failing repro + measured overhead, not adjectives.

## Rules

1. No silent plaintext fallback. Unsupported computation raises
   `UnsupportedComputation` (`src/exlex/circuit.py`, `src/exlex/backends/fhe.py`).
2. Every `PROTECTED` claim names its residual. A `PARTIAL` without a
   residual fails review (`src/exlex/concerns.py`, `src/exlex/report.py`).
3. Run `ruff check src tests`, `exlex audit src/ --fail-on high`, and the
   relevant `bench/*` + `examples/*` before opening a PR.
4. Security reports: open a public issue per `SECURITY.md`. No private channel.
