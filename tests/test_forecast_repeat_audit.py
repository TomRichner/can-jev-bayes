import pytest

from jevbandits.forecast_repeat_audit import decompose


def test_identical_biased_repeats_have_zero_variability():
    result = decompose([0.6, 0.4], [0.6, 0.4], [0.5, 0.5])
    assert result["raw_excess_brier"] == pytest.approx(0.02)
    assert result["repeat_variability"] == 0
    assert result["cross_repeat_error_product"] == pytest.approx(0.02)


def test_opposite_errors_are_not_clipped():
    result = decompose([0.6, 0.4], [0.4, 0.6], [0.5, 0.5])
    assert result["raw_excess_brier"] == pytest.approx(0.02)
    assert result["repeat_variability"] == pytest.approx(0.04)
    assert result["cross_repeat_error_product"] == pytest.approx(-0.02)
