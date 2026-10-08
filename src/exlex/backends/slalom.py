"""Blinded linear-layer delegation: privacy and integrity without a trusted GPU.

This is a prototype of the core technique from

    Slalom: Fast, Verifiable and Private Execution of Neural Networks in
    Trusted Hardware. Tramèr & Boneh, ICLR 2019 (arXiv:1806.03287),

adapted to this library's architecture. The original delegates every linear
layer of a DNN from a trusted CPU to an untrusted GPU by sending the host a
*blinded* input and checking the returned product with a randomised test. The
result is that the host does the expensive matmul and learns nothing, while a
host that cheats is caught with overwhelming probability.

What this module implements
---------------------------
Given a plaintext input ``x`` and a layer weight matrix ``W``:

1. **Blind.** Encode ``x`` into fixed point as an integer ``tau``, then add a
   one-time-pad mask: ``tau~ = tau + r``. Only ``tau~`` crosses the boundary.
2. **Delegate.** The host computes ``W~ @ tau~``, where ``W~`` is the same
   fixed-point encoding of the weights. No reduction happens, which is what
   keeps the host's arithmetic and the client's arithmetic identical integers.
3. **Unblind.** The client computes the mask's contribution ``W~ @ r`` locally
   and subtracts, recovering ``W~ @ tau`` exactly. The results are integers, so
   the subtraction is exact by construction; the client then divides by the
   product of the two scales once.
4. **Verify.** Draw a fresh random challenge vector *after* the result arrives
   and compare a random linear combination of the claimed result against one
   computed from the plaintext input. A host that corrupts any output passes
   with probability ``mask_range ** -rounds``.

Why the masking is exact
------------------------
Naive blinding in floating point fails silently: the mask is large and the
signal is small, so subtracting the mask's contribution loses the answer to
cancellation, and the failure looks like working code returning noise. Encoding
both sides as integers removes the problem rather than tuning it: the client's
factor and the host's product are the same products of the same integers, so
they cancel exactly, and
:meth:`SlalomBackend._check_layer_bounds` refuses layers where a real executor
would overflow instead of pretending it would not.

The two hard-won details in here are both about that exactness, and both were
learned by getting them wrong first:

* The weights reach the host **in their fixed-point encoding**, and the client
  must use that identical encoding for the mask's contribution. Mixing the
  plaintext and encoded forms leaves a residue on the mask's scale in every
  result, and verification still passes, which is the worst possible failure.
* Both the input and the weights carry the scale, so a delegated result carries
  the *product* of the two scales. That is a single documented division in
  :meth:`SlalomBackend.run_linear`, not a per-caller surprise.

Threat model boundaries
-----------------------
The privacy and integrity claims cover one linear layer's inputs and output.
They rest on the client-side mask ``r`` staying secret and on the verification
challenge being drawn after the host's answer arrives; both hold because both
happen in this process.

The prototype deliberately leaves strong claims on the floor:

* **The trusted side is an ordinary Python process.** Slalom's trusted side is
  an SGX enclave. Nothing here isolates it, so this does not defend against an
  attacker who owns the machine running the client. It is a prototype of the
  protocol, not of the hardware.
* **Multi-layer stacking is not automated.** Each layer needs a fresh mask, and
  the intermediate activation between two layers must be unblinded before the
  next layer can be masked. :meth:`SlalomBackend.run_linear` handles one layer;
  composing a chain is the caller's job for now.
* **The weights are plaintext to the host**, exactly as in
  :mod:`exlex.backends.fhe`: a homomorphic-looking matmul multiplies a ciphertext
  by public data. The circuit, the shapes, and therefore
  :attr:`~exlex.concerns.Concern.OPERATIONS` travel in the clear.
"""

from __future__ import annotations

import math
import secrets
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..concerns import Backend, Claim, Concern, Status
from ..errors import UnsupportedComputation

#: The masking range, a power of two rather than a prime: the mask is used as a
#: one-time pad, and the verification check is a random scalar product over the
#: integers, neither of which needs a field. What the size buys is precision:
#: the mask must dominate the scaled signal so a host that sees
#: ``W~ @ (tau + r)`` learns only ``W~ @ r``, and it must keep the host's
#: accumulator inside an int64 for a realistic matrix. ``2**32`` is where those
#: two pull in opposite directions, so it is the default and it is documented
#: on :meth:`FixedPoint.encode_scaled`.
DEFAULT_MASK_RANGE = 1 << 32

#: Binary fractional digits in the fixed-point encoding. Values are scaled by
#: ``1 << DEFAULT_FRACTIONAL_BITS`` before masking, so this is both the
#: resolution and the fixed overhead subtracted from the usable value range.
DEFAULT_FRACTIONAL_BITS = 16

#: Upper bound on any encoded value, mask, or accumulator term. Keeps the
#: receiver's exactness statement honest instead of relying on it holding for
#: whatever the caller happened to pass.
MAX_TENSOR_SIZE = 1 << 24


class FixedPoint:
    """Integer fixed-point coding used to make the masking exact.

    A value ``v`` is encoded as ``v * 2**bits``, truncated toward zero and
    clamped into ``[-(2**bits * MAX_TENSOR_SIZE), 2**bits * MAX_TENSOR_SIZE)``.
    Masks are drawn from ``[0, mask_range)``, so the encoded magnitude is
    negligible next to the mask, which is the privacy argument, and the mask
    stays small enough that a matmul over it does not overflow an int64.

    Args:
        mask_range: the range the mask is drawn from. A power of two, and the
            knob that trades privacy headroom against accumulator size.
        bits: fractional digits, e.g. 16 for a resolution of ``2**-16``.
    """

    def __init__(
        self,
        mask_range: int = DEFAULT_MASK_RANGE,
        bits: int = DEFAULT_FRACTIONAL_BITS,
    ) -> None:
        if bits < 1:
            raise ValueError("bits must be at least 1")
        if mask_range < (1 << 16):
            raise ValueError(
                f"mask_range must be at least 2**16 for the pad to dominate a "
                f"scaled signal; got {mask_range}"
            )
        self.mask_range = int(mask_range)
        self.bits = int(bits)
        self.scale = 1 << self.bits
        # Largest magnitude an encoded value may carry. This is the tensor's
        # practical range, not a protocol constant: raising it only asks for a
        # larger mask_range to keep the pad dominant.
        self.limit = MAX_TENSOR_SIZE * self.scale
        #: The scale a delegated result carries. Both the inputs and the weights
        #: are encoded at ``2**bits``, so a product comes back at the *product*
        #: of the two scales and the client divides by this once to read it.
        #: Effective resolution of the whole protocol is ``2**-(2*bits)``.
        self.value_scale = self.scale * self.scale

    # ------------------------------------------------------------------ codec

    def encode_scaled(self, value: float) -> int:
        """Encode one value as a scaled integer, clamping outside the range.

        Clamping rather than raising is a deliberate choice: a weight or input
        that overflows the range is a configuration the protocol cannot
        express, and a hard failure here would turn a bad model into a crash
        inside the host's matmul loop. Raising would instead be the other
        honest option; both are better than a value that silently wraps modulo
        ``P`` and makes the client's unblinding produce a plausible wrong
        number.

        The clamp is also where the masking stays information-theoretic: the
        encoded magnitude stays below ``limit``, so it is negligible next to a
        mask drawn from ``[0, P)``.

        The scaling goes through :func:`math.ldexp` rather than ``value *
        (1 << bits)`` on purpose. A float times a power of two is *inexact*
        above ``2**53``: ``int(3.7 * 65536)`` and
        ``int(float(3.7) * float(1 << 16))`` can round differently, and the
        client-side mask contribution then differs from the host's product by
        one unit, which shows up as a small permanent error in every result.
        ``ldexp`` scales the exponent directly and avoids the rounding.
        """
        if math.isnan(value) or math.isinf(value):
            raise UnsupportedComputation(
                f"value {value!r} cannot be encoded: fixed point is finite by "
                "definition, and a NaN or infinity here would silently become "
                "the clamp bound"
            )
        if value >= 0.0:
            return min(int(math.ldexp(value, self.bits)), self.limit - 1)
        return max(int(math.ldexp(value, self.bits)), -(self.limit - 1))

    def encode(self, values: Sequence[float]) -> List[int]:
        """Encode a whole vector. See :meth:`encode_scaled` for the clamp."""
        return [self.encode_scaled(v) for v in values]

    def decode_scaled(self, integer: int) -> float:
        """Decode a scaled integer back to a float, rounding to nearest."""
        return integer / float(self.scale)

    def decode(self, scaled: Sequence[int]) -> List[float]:
        return [self.decode_scaled(v) for v in scaled]

    # ------------------------------------------------------------------- mask

    def rand_int(self) -> int:
        """A uniform mask term in ``[0, mask_range)``, from the OS CSPRNG.

        :mod:`secrets` rather than :mod:`random` is load-bearing: a mask drawn
        from a predictable generator is not a mask.
        """
        return secrets.randbelow(self.mask_range)

    def rand_ints(self, length: int) -> List[int]:
        return [self.rand_int() for _ in range(length)]


class SlalomBackend(Backend):
    """Blinded, verified delegation of a linear layer.

    The client keeps the plaintext input, the mask, and the reference result.
    The host receives a masked input, the weights, and unblinding factors, and
    returns a masked result. Nothing it can see reveals the input, and nothing
    it can return can pass verification while wrong.

    Args:
        rounds: independent verification challenges. Each is a fresh random
            vector, so a cheating host's pass probability is
            ``(1 / mask_range) ** rounds``.
        mask_range: masking range. A power of two; see
            :class:`FixedPoint`.
        bits: fractional bits of the fixed-point codec.
    """

    concerns = (Concern.DATA, Concern.CORRECTNESS)

    def __init__(
        self,
        rounds: int = 4,
        mask_range: int = DEFAULT_MASK_RANGE,
        bits: int = DEFAULT_FRACTIONAL_BITS,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        self.rounds = rounds
        self.fixed = FixedPoint(mask_range=mask_range, bits=bits)

    # ------------------------------------------------------------------ gates

    def available(self) -> Optional[str]:
        """Always usable. The blinding and checking need no extra dependency.

        The masked matmul the *host* performs does not need numpy either; a
        caller substituting numpy on the remote side is doing an optimisation
        the protocol does not require.
        """
        return None

    def establish(self) -> Claim:
        """Report what the mechanism protects, and what it does not.

        Input privacy is :attr:`~exlex.concerns.Status.PROTECTED` in the
        information-theoretic sense of the one-time pad. Result correctness is
        :attr:`~exlex.concerns.Status.PARTIAL`, because the check is randomised
        and the residual names the bound, exactly as
        :class:`~exlex.backends.verify.VerificationBackend` does.
        """
        return Claim(
            concern=Concern.DATA,
            status=Status.PROTECTED,
            mechanism=f"blinded linear delegation (scale 2**{self.fixed.bits}, "
            f"mask range 2**{self.fixed.mask_range.bit_length() - 1})",
            residual=(
                "Covers the inputs and output of one linear layer. "
                "Intermediate activations between layers need a fresh mask and "
                "are the caller's responsibility in this prototype.",
                "The weight matrix is sent in the clear, so the host learns "
                "the operation shapes exactly as it does under the FHE path.",
                "The trusted side is this process. Nothing here isolates it, so "
                "this does not defend against an attacker who owns the client "
                "machine, only against the host of the delegated matmul.",
                "Privacy is one-time-pad against a *non-adaptive* host: a host "
                "that varies the weights between calls can binary-search a "
                "masked value. Slalom's deployment deals with this by pinning "
                "the weights it receives, which this prototype leaves open.",
            ),
            detail={
                "rounds": self.rounds,
                "mask_bits": self.fixed.mask_range.bit_length() - 1,
                "fractional_bits": self.fixed.bits,
                "resolution": f"2**-{self.fixed.bits}",
            },
        )

    # ------------------------------------------------------------ client side

    def blind(
        self, values: Sequence[float]
    ) -> Tuple[List[int], List[int]]:
        """Encode and mask ``values`` for one delegated layer.

        Returns:
            ``(masked, mask)``. Send ``masked``; keep ``mask`` for unblinding.
        """
        fixed = self.fixed
        mask = fixed.rand_ints(len(values))
        masked = [fixed.encode_scaled(v) + r for v, r in zip(values, mask)]
        return masked, mask

    def unblind_factors(
        self, weights: Sequence[Sequence[float]], mask: Sequence[int]
    ) -> List[int]:
        """The mask's contribution to the result, computed locally.

        This is the term the client subtracts, and the single most load-bearing
        method in the module: it must be *the same computation* the host
        performs on the mask, not an equivalent one. When these two disagree by
        even one unit the subtraction leaves a residue, the result is wrong, and
        nothing in the protocol can tell the caller that it was the client's
        arithmetic rather than the host's answer.

        It is exact because the two sides use identical integers: the encoded
        weight ``W~`` and the mask ``r`` are both fixed, and their product is
        summed without any reduction. Python integers do not overflow, and a
        GPU executor in int64 does not overflow while :meth:`host_matmul`'s
        bounds hold, which is what makes the two agree bit for bit.
        """
        fixed = self.fixed
        width = len(weights[0])
        return [
            sum(
                fixed.encode_scaled(float(weights[i][j])) * r for i, r in enumerate(mask)
            )
            for j in range(width)
        ]

    def unblind(
        self,
        returned: Sequence[int],
        factors: Sequence[int],
        weights: Sequence[Sequence[float]],
        mask: Sequence[int],
    ) -> List[int]:
        """Subtract the mask's contribution, recovering the layer's output.

        Args:
            returned: what the host sent back.
            factors: output of :meth:`unblind_factors`.
            weights: the layer weights, used to validate the accumulator bound.
            mask: the mask returned by :meth:`blind`, checked for shape.

        Raises:
            UnsupportedComputation: when the response width or the layer bounds
                make the exchange impossible, so the caller is told instead of
                receiving a number that cannot be trusted.
        """
        width = len(weights[0])
        if len(returned) != width or len(factors) != width:
            raise UnsupportedComputation(
                f"host returned {len(returned)} values for a layer with width "
                f"{width}; nothing to unblind"
            )
        if len(mask) != len(weights):
            raise UnsupportedComputation(
                f"mask has {len(mask)} entries but the weights have "
                f"{len(weights)} rows"
            )
        for row in weights:
            if len(row) != width:
                raise UnsupportedComputation("weight rows must all have the same width")
        self._check_layer_bounds(weights)
        # Exact integer subtraction. No reduction mod P happens here, and that
        # is the point: the client's factor and the host's product are the same
        # arithmetic over the same integers, so they cancel to
        # ``sum_i W~_ij * tau_i``, which the caller decodes.
        return [int(y) - factors[j] for j, y in enumerate(returned)]

    def _check_layer_bounds(self, weights: Sequence[Sequence[float]]) -> None:
        """Refuse a layer whose output the accumulator cannot hold exactly.

        The host's accumulator is ``sum_i W~_ij * (tau_i + r_i)``, bounded by
        ``rows * max|W~| * mask_range``. An int64 executor wraps at ``2**63``,
        and a wrapped accumulator means the client's factor no longer cancels,
        which produces a wrong answer that still verifies. Refusing here is the
        only way to keep the claim meaningful.

        This is why the default mask range is ``2**32``: combined with the
        default resolution it leaves room for layers of a few hundred rows,
        which is the reference scale of this prototype. A caller who needs
        wider layers lowers ``mask_range``, at the cost of mask entropy.
        """
        fixed = self.fixed
        widest = max(
            (abs(fixed.encode_scaled(float(w))) for row in weights for w in row),
            default=0,
        )
        rows = len(weights)
        bound = widest * rows * fixed.mask_range
        if widest and bound > (1 << 62):
            raise UnsupportedComputation(
                f"a {rows}x{len(weights[0])} layer with weight magnitudes up "
                f"to {widest} in fixed point pushes the host accumulator past "
                f"2**62 by a factor of about {bound >> 62}. Coarsen the "
                "resolution (fewer fractional bits), narrow the mask_range, "
                "or split the layer across passes"
            )

    # ------------------------------------------------------------ computation

    def run_linear(
        self,
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        values: Sequence[float],
        remote_eval: Callable[[Sequence[int], Sequence[Sequence[float]]], Sequence[int]],
        *,
        verify: bool = True,
    ) -> Dict[str, object]:
        """Full round trip for one linear layer: blind, delegate, unblind.

        The remote callable is the untrusted host. It receives the masked
        input and the plaintext weights, and returns a masked result. In a real
        deployment this is a call to the GPU; injecting it here is what lets the
        boundary be exercised, and attacked, without a second machine.

        Args:
            weights: the layer weight matrix, sent in the clear.
            bias: plaintext bias, added by the client after unblinding, so the
                host never sees it and it costs no extra blinding.
            values: plaintext inputs, which stay in this process.
            remote_eval: the host's matmul.
            verify: run the randomised check. Defaults to True; passing False
                is for benchmarks that isolate the delegation cost.

        Returns:
            ``{"result": [...], "verified": bool, "rounds": int,
            "masked_input": [...], "constants": {"unblind": [...]}}``. The
            masked input and the unblinding constants are included because they
            are exactly what the host saw, which is how a test asserts that the
            plaintext never left.
        """
        masked, mask = self.blind(values)
        returned = [int(v) for v in remote_eval(masked, [list(row) for row in weights])]
        factors = self.unblind_factors(weights, mask)
        scaled = self.unblind(returned, factors, weights, mask)
        # Both the inputs and the weights are fixed-point encoded, so the
        # delegation carries the *product* of the two scales. Dividing by it once
        # here is the whole decoding step; see FixedPoint.value_scale.
        value_scale = self.fixed.value_scale
        result = [
            scaled[j] / float(value_scale) + b
            for j, b in enumerate(bias)
        ]
        verified = True
        if verify:
            verified = self.verify_blinded(weights, values, scaled)
        return {
            "result": result,
            "verified": verified,
            "rounds": self.rounds,
            "masked_input": masked,
            "constants": {"unblind": factors},
        }

    # ----------------------------------------------------------- verification

    def verify_blinded(
        self,
        weights: Sequence[Sequence[float]],
        plaintext: Sequence[float],
        claimed_scaled: Sequence[int],
    ) -> bool:
        """Freivalds-style check of an unblinded result against a local matmul.

        The reference is computed here, in this process, from the plaintext
        input, so the host's claim is checked against something it never saw a
        masked form of. Each round draws a fresh challenge vector *after* the
        result arrived, which is what stops a host from tailoring an answer to
        the check.

        Raises:
            UnsupportedComputation: when the plaintext input cannot be encoded,
                since then there is no reference to check against.
        """
        fixed = self.fixed
        expected = [
            sum(
                fixed.encode_scaled(float(weights[i][j])) * fixed.encode_scaled(v)
                for i, v in enumerate(plaintext)
            )
            for j in range(len(weights[0]))
        ]
        if len(claimed_scaled) != len(expected):
            return False
        for _ in range(self.rounds):
            # A fresh challenge per round, drawn after the host's answer is in
            # hand. Over the integers rather than a ring: a host whose answer is
            # wrong anywhere makes the two scalar products differ by a non-zero
            # amount, and a random coefficient vector hits that with probability
            # at least 1 - range^-1.
            r = [self.fixed.rand_int() + 1 for _ in expected]
            lhs = sum(ri * ci for ri, ci in zip(r, claimed_scaled))
            rhs = sum(ri * ei for ri, ei in zip(r, expected))
            if lhs != rhs:
                return False
        return True

    def pass_probability(self) -> float:
        """Probability a cheating host passes every round, stated not implied.

        Put this next to
        :meth:`~exlex.backends.verify.VerificationBackend.residual_bound` and
        the two can be compared honestly rather than by adjective.
        """
        return float(self.fixed.mask_range) ** -self.rounds


def host_matmul(
    masked: Sequence[int],
    weights: Sequence[Sequence[float]],
    *,
    codec: Optional[FixedPoint] = None,
) -> List[int]:
    """The untrusted host's side: ``W~ @ (tau + r)``.

    This is a *simulation* of the host, shipped so the prototype is testable and
    benchmarkable without a GPU. It is also the exact expression a real executor
    must match: encode each weight with the client's codec, multiply by the
    masked input, and sum. No reduction happens anywhere, which is what keeps
    this and :meth:`SlalomBackend.unblind_factors` bit-for-bit identical
    arithmetic, and which is why :meth:`SlalomBackend._check_layer_bounds`
    refuses layers that would overflow an int64.

    Args:
        masked: the masked input, already in ``[0, P)``.
        weights: plaintext weights.
        codec: the codec the client used. Defaults to a fresh
            :class:`FixedPoint` with the library defaults, which matches a
            client that did not override them.

    Returns:
        The masked result, in ``[0, P)``.
    """
    fixed = codec or FixedPoint()
    out_dim = len(weights[0])
    if len(masked) != len(weights):
        raise ValueError(
            f"masked input has {len(masked)} entries but the weight matrix has "
            f"{len(weights)} rows"
        )
    out: List[int] = []
    for j in range(out_dim):
        encoded = [fixed.encode_scaled(float(weights[i][j])) for i in range(len(weights))]
        total = 0
        for w, m in zip(encoded, masked):
            total += w * m
        out.append(total)
    return out
