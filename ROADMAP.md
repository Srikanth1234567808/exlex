# Roadmap: from defence-in-depth to useful untrusted compute

Question: can we run useful ML on a GPU host we do not control,
at a cost someone would actually pay?

## M1 — Data-private baseline (done)

* `FHEBackend`: CKKS encrypt → remote matmul → local decrypt (`src/exlex/backends/fhe.py`).
* `Circuit`: compile affine + polynomial-approx layers, refuse the rest (`src/exlex/circuit.py`).
* `VerificationBackend`: randomised Freivalds check (`src/exlex/backends/verify.py`).
* `SlalomBackend`: blinded one-layer delegation prototype (`src/exlex/backends/slalom.py`).
* Report honestly: `OPERATIONS` unprotected, `CORRECTNESS` partial (`src/exlex/report.py`).

Acceptance: `bench/encrypted_pipeline.py` round-trips and catches a corrupted result.

## M2 — Ops-obscuring prototype (next, good-first-issues)

Goal: host sees fixed-width, fixed-depth work, not the real architecture.

* Strip names, pad every `AffineLayer` to a max `(in, out)`, pad depth with no-ops.
* Show `Circuit.describe()` before/after as the leak demo.
* New backend claiming `OPERATIONS: PARTIAL` with residual: total padded
  work + timing still visible.
* Attack: fingerprint the model by timing/size despite padding.

Acceptance: padded circuit evaluates correctly; report shows what still leaks;
overhead measured in `bench/`.

## M3 — Verifiable delegation at useful cost

* Compose per-layer Slalom checks; multi-layer masking without caller plumbing.
* Real ZK option via `ZKBackend(verify=...)`; state cost vs randomised check.
* End-to-end: encrypted + verified pipeline under a stated Nx budget.

Acceptance: dishonest host caught with stated bound; overhead published, not claimed.

## Non-goals

1x overhead, hiding metadata (sizes/timing), stopping result-withholding.
See `SECURITY.md` and `exlex doctor` residuals.
