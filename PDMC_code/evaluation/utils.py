import numpy as np


def get_delta(factual: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """
    Compute difference between original factual and counterfactual

    Parameters
    ----------
    factual: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    counterfactual: : np.ndarray
        Normalized and encoded array with counterfactual data.
        Shape: NxM

    Returns
    -------
    np.ndarray
    """
    return counterfactual - factual


def nni_target(qualified_rate: float, delta_x_kde_percentile: float, x_kde_percentile: float, validate_rate: float):
    results: list = [qualified_rate, delta_x_kde_percentile, x_kde_percentile]
    return float(np.mean(results)) * validate_rate

def nni_target_v2(qualified_rate: float, delta_x_kde_percentile: float, x_kde_percentile: float, validate_rate: float):
    result: float = qualified_rate * delta_x_kde_percentile * (x_kde_percentile ** 0.25)
    result += (10 * (validate_rate - 1.0))
    return result


def nni_target_v3(qualified_rate: float, delta_x_kde_percentile: float, x_kde_percentile: float, validate_rate: float):
    result: float = qualified_rate * delta_x_kde_percentile * x_kde_percentile
    result += (10 * (validate_rate - 1.0))
    return result

def nni_target_v4(delta_x_kde_percentile: float, x_kde_percentile: float, validate_rate: float):
    result: float = delta_x_kde_percentile * x_kde_percentile
    result += (10 * (validate_rate - 1.0))
    return result
