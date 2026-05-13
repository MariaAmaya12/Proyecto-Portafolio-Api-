import numpy as np
import pytest

from src.fixed_income import (
    bond_price,
    convexity,
    macaulay_duration,
    modified_duration,
    nelson_siegel_yield,
)


def test_bond_price_with_coupon():
    price = bond_price(1000, 0.05, 0.04, 2, frequency=1)

    assert price == pytest.approx(1018.8609467455621)


def test_macaulay_duration_positive():
    assert macaulay_duration(1000, 0.05, 0.04, 2) > 0


def test_modified_duration_less_than_macaulay():
    macaulay = macaulay_duration(1000, 0.05, 0.04, 5)
    modified = modified_duration(1000, 0.05, 0.04, 5)

    assert modified < macaulay


def test_convexity_positive():
    assert convexity(1000, 0.05, 0.04, 5) > 0


def test_nelson_siegel_float_and_array():
    scalar = nelson_siegel_yield(2.0, 0.04, -0.02, 0.01, 1.5)
    array = nelson_siegel_yield(np.array([1.0, 2.0]), 0.04, -0.02, 0.01, 1.5)

    assert isinstance(scalar, float)
    assert array.shape == (2,)
