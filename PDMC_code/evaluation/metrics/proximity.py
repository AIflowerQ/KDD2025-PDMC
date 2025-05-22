import numpy as np
import pandas as pd
import torch

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX
from evaluation.utils import get_delta
from predict_models.api.predict_models import MLModel
from utils.utils import get_torch_model_device



def l0_distance(delta: np.ndarray) -> np.array:
    """
    Computes L-0 norm, number of non-zero entries.

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    1-d np.array
    """
    # get mask that selects all elements that are NOT zero (with some small tolerance)
    difference_mask = np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05))
    # get the number of changed features for each row
    num_feature_changes = np.sum(
        difference_mask,
        axis=1,
        dtype=np.float,
    )
    distance = num_feature_changes.reshape(-1)
    return distance


def l1_distance(delta: np.ndarray) -> np.array:
    """
    Computes L-1 distance, sum of absolute difference.

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    1-d np.array
    """
    absolute_difference = np.abs(delta)
    distance = np.sum(absolute_difference, axis=1, dtype=np.float).reshape(-1)
    return distance


def l2_distance(delta: np.ndarray) -> np.array:
    """
    Computes L-2 distance, sum of squared difference.

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    1-d np.array
    """
    squared_difference = np.square(np.abs(delta))
    distance = np.sum(squared_difference, axis=1, dtype=np.float).reshape(-1)
    return distance


def linf_distance(delta: np.ndarray) -> np.array:
    """
    Computes L-infinity norm, the largest change

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    1-d np.array
    """
    absolute_difference = np.abs(delta)
    # get the largest change per row
    largest_difference = np.max(absolute_difference, axis=1)
    distance = largest_difference.reshape(-1)
    return distance


def calculate_mad_each_variable(training_data: np.array) -> np.array:
    """
    :param training_data: 2-d
    :return: 1-d
    """
    medians = np.median(training_data, axis=0)
    absolute_deviation = np.abs(training_data - medians)
    mad_values = np.median(absolute_deviation, axis=0)
    return mad_values

def distance_mad(factuals: np.ndarray, counterfactuals: np.ndarray, mad_values: np.ndarray) -> np.array:
    absolute_differences = np.abs(counterfactuals - factuals)

    # print(absolute_differences)
    # print(mad_values)

    safe_mad_values: np.ndarray = np.copy(mad_values)
    safe_mad_values[mad_values == 0] = 1.0
    # print(safe_mad_values)

    normalized_differences = absolute_differences / safe_mad_values
    # print(normalized_differences)
    distances = np.mean(normalized_differences, axis=1)
    return distances


class ProximityEvaluation(EvaluationIndices):
    """
    Computes success rate for the whole recourse method.
    """

    NAME: str = 'Proximity'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array):

        delta: np.array = get_delta(valid_factuals, valid_counterfactuals)

        l2: np.array = l2_distance(delta)

        l1_mad: np.array = distance_mad(valid_factuals, valid_counterfactuals, self.mad_values)

        return {
            'l2': l2,
            'l1_mad': l1_mad
        }


    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def __init__(self, data_manager: DataCatalog, valid_indices: np.array):
        metric_name_list: list = ['l2', 'l1_mad']
        default_dict: dict = {
            'l2': DEFAULT_MAX,
            'l1_mad': DEFAULT_MAX
        }
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)

        training_data: np.array = self.data_manager.df[self.data_manager.feature_columns_order].values
        self.mad_values: np.array = calculate_mad_each_variable(training_data)




