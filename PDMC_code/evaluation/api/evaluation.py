from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from data.catalog.catalog import DataCatalog


DEFAULT_MAX = 1e3
DEFAULT_MIN = -DEFAULT_MAX
DEFAULT_ZERO = 0.


class Evaluation(ABC):
    def __init__(self, data_manager: DataCatalog):
        self.data_manager: DataCatalog = data_manager

    @abstractmethod
    def get_evaluation(
        self, factuals: np.array, counterfactuals: np.array
    ) -> (dict, dict):
        """Compute evaluation measure"""
        pass


class EvaluationIndices(Evaluation, ABC):
    NAME: str = 'Abstract'
    def __init__(
            self, data_manager: DataCatalog,
            valid_indices: np.array,
            metric_name_list: list,
            default_number_dict: dict = None,
            use_invalid_process: bool = False,
    ):
        super().__init__(data_manager)

        self._valid_indices: np.array = valid_indices

        self._use_invalid_process: bool = use_invalid_process

        self._default_number_dict: dict = default_number_dict

        self.metric_name_list: list = metric_name_list

        assert len(set(metric_name_list)) == len(metric_name_list)

        assert not (default_number_dict is None and use_invalid_process is False)
        assert not (default_number_dict is not None and use_invalid_process is not False)

        if default_number_dict is not None:
            assert set(default_number_dict.keys()) == set(self.metric_name_list)

    @property
    def use_invalid_process(self):
        return self._use_invalid_process

    @property
    def default_number_dict(self):
        return self._default_number_dict

    @property
    def valid_indices(self):
        return np.copy(self._valid_indices)

    @valid_indices.setter
    def valid_indices(self, valid_indices: np.array):
        assert len(valid_indices.shape) == 1
        self._valid_indices = valid_indices

    def _split_valid_and_invalid(self, two_dimension_arr: np.array):

        valid_rows: np.array = two_dimension_arr[self.valid_indices]
        invalid_rows: np.array = two_dimension_arr[~self.valid_indices]

        # exit()

        return valid_rows, invalid_rows

    def get_evaluation(
        self, factuals: np.array, counterfactuals: np.array
    ) -> (dict, dict):

        valid_factuals, invalid_factuals = self._split_valid_and_invalid(factuals)

        valid_counterfactuals, invalid_counterfactuals = self._split_valid_and_invalid(counterfactuals)

        if len(valid_factuals) == 0 and self.use_invalid_process is False:
            # print('23333')
            return {}, self.default_number_dict

        valid_results_dict: dict = self.valid_process(valid_factuals, valid_counterfactuals)

        results_mean_dict: dict = dict()
        results_dict: dict = dict()
        if self.use_invalid_process is False:
            for metric in self.metric_name_list:
                results_mean_dict[metric] = float(np.mean(valid_results_dict[metric]))
                results_dict[metric] = valid_results_dict[metric]
            return results_mean_dict, results_dict

        else:
            invalid_results_dict: dict = self.invalid_process(invalid_factuals, invalid_counterfactuals)
            for metric in invalid_results_dict.keys():
                all_results: np.array = np.concatenate((valid_results_dict[metric], invalid_results_dict[metric]))
                results_mean_dict[metric] = float(np.mean(all_results))
                results_dict[metric] = all_results
            return results_mean_dict, results_dict




    @abstractmethod
    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array) -> dict:
        """
        :param valid_factuals:
        :param valid_counterfactuals:
        :return: dict of (1-d np array for each sample)
        """
        pass

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        """
        :param invalid_factuals:
        :param invalid_counterfactuals:
        :return: dict of (1-d np array for each sample)
        """
        raise NotImplementedError




