# Security policy

## Scope, stated first

`exlex` is defence-in-depth. It is not a TEE, it does not encrypt GPU memory,
and it does not make a consumer 3090 or 4090 safe from a host that owns the
machine. Before reporting something, check whether it falls in the list of
things this project has already documented as not protecting you:

- `exlex levels` prints the residual risk at every assurance level.
- `exlex protect` prints, per concern, what is protected and what is not.
- `README.md` has a "Known blind spots" section.

Reporting "the host can still read VRAM on a 4090" is a report about hardware,
not about this code, and it is already answered.

## What is in scope

A vulnerability in this library means code that does not do what it reports.
Concretely:

- A claim of `PROTECTED` that a host can defeat, for example a context that
  turns out to retain a usable secret key.
- A key, plaintext input, or model weight that reaches a remote, log, or
  exception message.
- A backend that silently falls back to plaintext instead of raising
  `UnsupportedComputation`.
- Verification that passes a result it should reject.
- The `exlex audit` rules missing a crossing they claim to cover.
- Anything in the `attest` path that treats a failure as a pass.

## What is out of scope

- Host operator capabilities: root on the machine, `/proc/kcore`,
  `virsh dump`, hypervisor DMA, physical access.
- Unencrypted GPU VRAM on consumer parts.
- Side channels, including timing, cache, and power analysis, except where a
  report identifies a specific mitigation we claimed and do not have.
- Metadata exposure: sizes, shapes, timing, and duration are documented as
  visible in every configuration.
- Denial of service by a host withholding or delaying a result.
- Weaknesses in TenSEAL, SEAL, or snpguest. Report those upstream. We will help
  reproduce.
- Models containing an unsupported layer. That raises by design.

## Reporting

Open a public issue. Do not send a private report.

This is a deliberate choice and the reasoning is worth stating, because it
differs from most security projects. A single-maintainer project cannot credibly
run a private disclosure channel: it cannot triage in secret, patch in secret,
or credit a reporter under a handle it has not verified, and an embargo it
cannot honour is worse than no embargo because it delays the fix while implying
a coordination that is not happening.

Open issues also work better for this particular project. A claim that a
`PROTECTED` status is wrong is checkable by anyone reading the report code, and
the composition rules are in `exlex/report.py` in about two hundred lines. If a
contributor is going to catch it, they will catch it faster than a maintainer
will.

If a report turns out to expose a live deployment, the reporter's judgement
about their own situation governs. Say so in the issue if you need to.

## No warranty

There is none, and any claim to the contrary in this repository is a bug. See
`LICENSE` and the "What this is, and what it is not" section of `README.md`.
