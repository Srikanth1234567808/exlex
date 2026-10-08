"""Circuit compilation: what it accepts, and what it must refuse.

The refusal tests matter more than the acceptance ones. A compiler that
silently drops an unsupported layer would make an encrypted run disagree with
a plaintext one, and the only symptom would be a wrong number in production.
"""

from __future__ import annotations

import pytest

import exlex
from exlex.circuit import AffineLayer, BiasLayer, Circuit, PolynomialLayer
from exlex.errors import MissingDependency, UnsupportedComputation


def test_plaintext_evaluation_matches_hand_computation():
    circuit = Circuit(
        layers=[
            AffineLayer(weights=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), name="w"),
            BiasLayer(bias=(0.5, 0.5), name="b"),
        ]
    )
    # 3 rows x 2 cols, so 3 inputs and 2 outputs. Column 0 is 1+0+3=4 and
    # column 1 is 0+2+3=5, each plus the 0.5 bias.
    assert circuit.evaluate_plaintext([1.0, 2.0, 3.0]) == pytest.approx([4.5, 5.5])


def test_bias_is_not_silently_dropped():
    """A missing bias changes the answer, so it must be a separate layer."""
    with_bias = Circuit(
        layers=[
            AffineLayer(weights=((1.0, 1.0), (1.0, 1.0))),
            BiasLayer(bias=(10.0, 10.0)),
        ]
    )
    without = Circuit(layers=[AffineLayer(weights=((1.0, 1.0), (1.0, 1.0)))])
    # 2 rows x 2 cols. Each column sums both inputs, so [1, 2] gives [3, 3].
    assert with_bias.evaluate_plaintext([1.0, 2.0]) == pytest.approx([13.0, 13.0])
    assert without.evaluate_plaintext([1.0, 2.0]) == pytest.approx([3.0, 3.0])


def test_polynomial_layer_evaluates_in_ascending_order():
    layer = PolynomialLayer(coefficients=(0.0, 1.0, 1.0))  # x + x^2
    circuit = Circuit(layers=[layer])
    assert circuit.evaluate_plaintext([2.0, 3.0]) == pytest.approx([6.0, 12.0])


def test_mismatched_vector_length_is_refused():
    circuit = Circuit(layers=[AffineLayer(weights=((1.0, 0.0), (0.0, 1.0)))])
    with pytest.raises(UnsupportedComputation, match="row count"):
        circuit.evaluate_plaintext([1.0, 2.0, 3.0])


def test_describe_names_every_layer():
    circuit = Circuit(
        layers=[
            AffineLayer(weights=((1.0, 0.0), (0.0, 1.0)), name="fc1"),
            BiasLayer(bias=(0.0, 0.0), name="fc1.bias"),
        ]
    )
    text = circuit.describe()
    assert "fc1" in text
    assert "Affine(2->2)" in text
    assert "Bias(2)" in text


def test_describe_surfaces_approximations():
    circuit = Circuit(
        layers=[PolynomialLayer(coefficients=(0.0, 1.0), approximates="ReLU")]
    )
    text = circuit.describe()
    assert "approximated" in text
    assert "ReLU" in text


def test_shapes_expose_the_leak_explicitly():
    circuit = Circuit(
        layers=[
            AffineLayer(weights=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), name="a"),
            AffineLayer(weights=((1.0,), (1.0,)), name="b"),
        ]
    )
    # 2 rows x 3 cols: input width 2, output width 3. A host reads exactly this.
    assert circuit.shapes() == [(2, 3), (2, 1)]


def test_input_and_output_dims_use_the_column_convention():
    layer = AffineLayer(weights=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    assert layer.input_dim == 2
    assert layer.output_dim == 3


def test_compile_model_requires_torch_or_explains():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(MissingDependency, match="torch"):
            exlex.compile_model(object())
        return
    pytest.skip("torch present; the missing-dependency path cannot be exercised")


def test_circuit_exports_are_public():
    for name in ("AffineLayer", "BiasLayer", "Circuit", "PolynomialLayer", "compile_model"):
        assert name in exlex.__all__
        assert hasattr(exlex, name)
