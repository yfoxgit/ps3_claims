import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)
    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    winsorizer.fit(X)
    X_transformed = winsorizer.transform(X)

    expected_min = np.quantile(X, lower_quantile)
    expected_max = np.quantile(X, upper_quantile)

    assert X_transformed.max() <= expected_max
    assert X_transformed.min() >= expected_min