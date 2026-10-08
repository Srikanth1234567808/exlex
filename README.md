# UNTRUSTED COMPUTE

> **The machine is the adversary. We want useful computation anyway.**

## The Problem

We increasingly outsource computation to machines we do not own.

Cloud GPUs.
Rented servers.
Consumer GPUs.
Third-party infrastructure.

But outsourcing computation usually requires us to trust the machine.

What if we remove that assumption completely?

Imagine this:

You have a secret algorithm and secret data.

You send them to a machine you **do not trust**.

The machine owner has:

- root access
- control over the operating system
- control over the drivers
- control over the runtime
- control over the hypervisor
- visibility into system resources
- the ability to observe and manipulate execution

Yet you still want the machine to perform the computation.

You do **not** want the machine owner to learn:

- your data
- your algorithm
- your parameters
- intermediate values
- meaningful execution structure
- useful side-channel information

And you want to know whether the computation was performed correctly.

### The machine itself is the adversary.

Can we make this practical?

---

## Why?

Secure computation already exists.

We have powerful ideas such as:

- Fully Homomorphic Encryption
- Multi-Party Computation
- Zero-Knowledge Proofs
- Verifiable Computation
- Trusted Execution Environments
- Oblivious computation
- Secure compilers
- Side-channel-resistant systems

But there is a fundamental problem:

### Secure computation is expensive.

A technique can be cryptographically elegant and still be practically useless if a computation that takes one second normally takes thousands or millions of seconds securely.

So the question we care about is not simply:

> **Can secure computation be done?**

It is:

> **Can we make computation on an untrusted machine cheap enough to actually use?**

---

## The Extreme Case

We want to investigate the hardest practical setting we can find.

### Commodity GPU

An RTX 4090/3090-class machine.

### Hostile owner

The machine owner is actively malicious.

### No trusted software stack

We do not assume the operating system, hypervisor, drivers, runtime, or surrounding software is trustworthy.

### Secret computation

The computation itself may be proprietary.

### Secret inputs

The input data may be confidential.

### Host observes execution

We assume the attacker can observe things such as:

- timing
- memory transfers
- allocation patterns
- kernel launches
- execution structure
- resource usage
- other observable side channels

### Integrity matters

The machine should not be able to silently return an incorrect result.

This is deliberately an extreme threat model.

The purpose is not to claim that every aspect is immediately achievable.

The purpose is to discover:

> **What is possible, what is expensive, and what is fundamentally impossible?**

---

## The Real Research Question

We are interested in the intersection of:

**Cryptography × GPU Systems × Compilers × Security × Verification**

Can we simultaneously reduce:

- information leakage
- trust assumptions
- tampering risk

while reducing:

- computation overhead
- communication overhead
- memory overhead
- latency
- implementation complexity

---

## A Possible Direction

One possibility is that we should **not secure every operation equally** —
protect only what must be protected, and run the rest normally.

This is only one hypothesis.

It may be completely wrong.

Other approaches may involve:

- FHE acceleration
- MPC
- ZK proofs
- hybrid cryptographic protocols
- oblivious algorithms
- compiler transformations
- GPU-specific cryptographic primitives
- randomized execution
- side-channel defenses
- verifiable execution
- combinations of several techniques

We want to investigate the entire design space.

---

## What We Are Not Assuming

We are **not** assuming that:

- FHE is the answer
- MPC is the answer
- ZK is the answer
- TEEs are the answer
- GPUs are inherently secure
- the problem is solvable
- the overhead can reach 1×
- a single technique will be sufficient

We want evidence.

If something is impossible under the threat model, we want to understand **why**.

If something works, we want to know **how far it can scale**.

If something is too expensive, we want to know **where the cost comes from**.

---

## What Success Looks Like

Success does not necessarily mean:

> "We built a perfectly secure machine."

A meaningful result could be any of these:

### 1. A practical system

A real computation can be performed on hostile commodity hardware with strong confidentiality and integrity guarantees at useful performance.

### 2. A major reduction in overhead

A known secure technique becomes dramatically cheaper through GPU acceleration, compilation, protocol design, or another approach.

### 3. A new architecture

A combination of existing techniques produces a substantially better security/performance tradeoff.

### 4. A compiler

Given an ordinary program, automatically identify sensitive computation and transform it into a cheaper secure form.

### 5. A fundamental limitation

We discover that some desired property cannot be achieved under a particular threat model.

That is also a valuable result.

---

## How We Work

We want this to be an **open research effort**.

### Attack first.

If you think a system leaks information, try to extract it.

### Measure everything.

Security without performance measurements is incomplete.

### Reproduce results.

Experiments should be reproducible whenever possible.

### Challenge assumptions.

A beautiful architecture is worthless if its threat model quietly assumes a trusted machine.

### Prefer evidence over opinions.

If something is faster, benchmark it.

If something leaks, demonstrate it.

If something is impossible, explain the limitation.

### Publish failures.

A failed approach can be more useful than a successful one if it tells us where the boundary lies.

---

## Research Areas

This problem sits across several fields.

### Cryptography

- FHE
- MPC
- ZK
- secure protocols
- oblivious computation
- cryptographic primitives

### GPU Systems

- CUDA
- GPU architecture
- scheduling
- memory systems
- kernel execution
- GPU acceleration

### Compilers

- program analysis
- data-flow analysis
- control-flow transformation
- automatic partitioning
- secure compilation
- domain-specific languages

### Security

- side channels
- traffic analysis
- fault attacks
- malicious hosts
- information leakage
- execution monitoring

### Verification

- verifiable computation
- proof systems
- correctness verification
- probabilistic verification

---

## The First Challenge

We want to begin with something concrete:

> **Can we execute a secret computation on a hostile commodity GPU while minimizing what the machine owner can learn and preventing silent tampering?**

We will start small.

We will define the threat model precisely.

We will build baselines.

We will measure leakage.

We will measure performance.

We will attack our own systems.

Then we will iterate.

---

## Who Is This For?

This project is for people who enjoy problems that sit between disciplines.

You might be a:

- cryptographer
- GPU engineer
- compiler researcher
- systems engineer
- security researcher
- formal methods researcher
- ZK researcher
- MPC researcher
- hardware researcher
- mathematician

You do not need to agree with the premise.

In fact, if you think the problem is impossible, **we want to hear your argument.**

---

## Join the Research

This is intentionally starting small.

There is no finished system.

There is no established architecture.

There is no claim that the problem is solved.

There is only a question:

> **Can we make useful computation possible when the machine itself cannot be trusted?**

If that question interests you, open an issue, start a discussion, propose an experiment, or challenge the threat model.

**The machine is the adversary.**

**Let's see how much computation we can take back from it.**
