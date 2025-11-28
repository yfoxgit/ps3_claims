"""Evaluation helpers for model predictions.

This module implements a small, well-documented utility to compute
weighted evaluation metrics (MSE, RMSE, MAE, bias and weighted R^2)
given predictions, actuals and an optional sample weight (exposure).

The function uses numpy for vectorized computation. Inputs may be
array-like (lists, tuples, numpy arrays).
"""
from typing import Optional, Dict
from typing import Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback if numpy missing
    np = None


def _ensure_numpy(x):
    """Convert array-like to numpy array of floats (or raise if numpy missing)."""
    if np is None:
        raise RuntimeError("numpy is required for evaluate_predictions")
    return np.asarray(x, dtype=float)


def evaluate_predictions(predictions, actuals, exposure: Optional[object] = None) -> Dict[str, float]:
    """Compute weighted evaluation metrics for predictions.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Observed/true values.
    exposure : array-like or None, optional
        Sample weights (exposure). If None, uniform weights are used.

    Returns
    -------
    dict
        Dictionary with keys: 'mse', 'rmse', 'mae', 'bias', 'relative_bias', 'r2', 'deviance'.

    Notes
    -----
    - All metrics are weighted by `exposure` (if provided). Weighted R^2
      is computed as 1 - SSR / SST where SSR and SST use the same weights.
    - If the total weight is zero an error is raised.
    """
    p = _ensure_numpy(predictions)
    a = _ensure_numpy(actuals)

    if p.shape != a.shape:
        raise ValueError("predictions and actuals must have the same shape")

    if exposure is None:
        w = np.ones_like(a, dtype=float)
    else:
        w = _ensure_numpy(exposure)
        if w.shape != a.shape:
            raise ValueError("exposure (sample weight) must have the same shape as actuals/predictions")

    if np.any(w < 0):
        raise ValueError("exposure (sample weight) must be non-negative")

    w_sum = np.sum(w)
    if w_sum == 0:
        raise ValueError("sum of weights must be positive")

    diff = p - a
    mse = float(np.sum(w * diff ** 2) / w_sum)
    rmse = float(np.sqrt(mse))
    mae = float(np.sum(w * np.abs(diff)) / w_sum)

    # Compute exposure-weighted means for predictions and actuals.
    # Define bias as the deviation of the weighted mean prediction from
    # the weighted mean actual (i.e. mean(pred) - mean(actual)). This
    # is equivalent to the weighted mean residual but is written here
    # explicitly to match the requested definition.
    p_mean = float(np.sum(w * p) / w_sum)
    y_mean = float(np.sum(w * a) / w_sum)
    bias = float(p_mean - y_mean)
    # Relative bias: bias scaled by the exposure-weighted mean of actuals.
    relative_bias = float(bias / y_mean) if y_mean != 0.0 else float("nan")
    ssr = float(np.sum(w * (a - p) ** 2))
    sst = float(np.sum(w * (a - y_mean) ** 2))
    r2 = 1.0 - ssr / sst if sst != 0.0 else float("nan")

    # Poisson deviance (exposure-weighted, averaged):
    # D = 2 * sum_i w_i * (y_i * log(y_i / mu_i) - (y_i - mu_i))
    # with the convention that y*log(y/mu)=0 when y==0.
    # We return the deviance averaged by the total weight (consistent with other metrics).
    # Note: this is the Poisson deviance; if you want Gamma or Tweedie deviance, we can add an option.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a == 0.0, 1.0, a / p)
        # safe_log = 0 where a == 0 else log(a/p)
        safe_log = np.where(a == 0.0, 0.0, np.log(ratio))
        dev_contrib = a * safe_log - (a - p)
        deviance = float(2.0 * np.sum(w * dev_contrib) / w_sum)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "relative_bias": relative_bias,
        "r2": r2,
        "deviance": deviance,
    }


def gini_lorenz(predictions, actuals, exposure: Optional[object] = None) -> float:
    """Compute the Gini coefficient from the Lorenz curve as in ps3_script.py.

    This follows the same steps as the `lorenz_curve` function in
    `analyses/ps3_script.py`: sort observations by increasing predicted
    risk, compute the cumulative share of total claim amount (actual * exposure),
    compute the area under that Lorenz curve and return Gini = 1 - 2*AUC.

    Parameters
    ----------
    predictions, actuals : array-like
        Predictions (used for ranking) and observed values (pure premiums).
    exposure : array-like or None
        Sample weights/exposure. If None, uniform weights are used.

    Returns
    -------
    float
        Gini coefficient (may be NaN if total claim amount is zero).
    """
    p = _ensure_numpy(predictions)
    a = _ensure_numpy(actuals)

    if p.shape != a.shape:
        raise ValueError("predictions and actuals must have the same shape")

    if exposure is None:
        w = np.ones_like(a, dtype=float)
    else:
        w = _ensure_numpy(exposure)
        if w.shape != a.shape:
            raise ValueError("exposure (sample weight) must have the same shape as actuals/predictions")

    # Total claim amount (weighted)
    total_claim = float(np.sum(w * a))
    if total_claim == 0.0:
        return float("nan")

    # Order by increasing predicted risk (same as ps3_script)
    ranking = np.argsort(p)
    ranked_exposure = w[ranking]
    ranked_pure_premium = a[ranking]

    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount = cumulated_claim_amount / cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0.0, 1.0, len(cumulated_claim_amount))

    # Area under curve (AUC) using trapezoidal rule, then Gini = 1 - 2*AUC
    auc = float(np.trapz(cumulated_claim_amount, cumulated_samples))
    gini = 1.0 - 2.0 * auc
    return gini


def metrics_df(predictions, actuals, exposure: Optional[object] = None, column_name: str = "value"):
    """Compute evaluation metrics and return them as a pandas DataFrame.

    The returned DataFrame has metric names as the index and a single column
    named `column_name` containing the metric values. This is convenient for
    reporting and for concatenating multiple models side-by-side.
    """
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required to produce a DataFrame")

    metrics = evaluate_predictions(predictions, actuals, exposure)
    # Add gini as well
    try:
        g = gini_lorenz(predictions, actuals, exposure)
    except Exception:
        g = float("nan")
    metrics["gini"] = g

    df = pd.DataFrame.from_dict(metrics, orient="index", columns=[column_name])
    return df


def compare_models(pred_a, pred_b, actuals, exposure: Optional[object] = None, names: Sequence[str] = ("model_a", "model_b")):
    """Compare two sets of predictions side-by-side.

    Returns a DataFrame with metric names as the index and one column per model.
    """
    try:
        import pandas as pd
    except Exception:
        raise RuntimeError("pandas is required to produce a DataFrame")

    df_a = metrics_df(pred_a, actuals, exposure, column_name=names[0])
    df_b = metrics_df(pred_b, actuals, exposure, column_name=names[1])
    combined = pd.concat([df_a, df_b], axis=1)
    return combined
