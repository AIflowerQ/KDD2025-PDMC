import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX, DEFAULT_ZERO
from evaluation.utils import get_delta
from predict_models.api.predict_models import MLModel
from utils.utils import get_torch_model_device


class UserImmutableEvaluation(EvaluationIndices):
    """
    Computes success rate for the whole recourse method.
    """

    NAME: str = 'User Immutable'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array) -> dict:

        assert self.data_manager.is_transformed

        delta_abs: np.array = np.abs(get_delta(valid_factuals, valid_counterfactuals))

        result_dict: dict = dict()

        for eps_thr in self.eps_thr_list:

            result_array: np.array = np.full(valid_factuals.shape[0], True, dtype=bool)

            for user_immutable in self.data_manager.user_immutables:


                immutable_indicators: np.array = self.data_manager.user_immutables_individual_truth_test[user_immutable]
                immutable_indicators = immutable_indicators[self.valid_indices]

                # print(immutable_indicators)
                # print(immutable_indicators.shape)
                # print(valid_factuals.shape)

                assert immutable_indicators.shape[0] == valid_factuals.shape[0]

                immutable_column_indexes: list = self.data_manager.locate_feature_in_encoded_ordered_list(user_immutable)

                var_delta: np.array = delta_abs[:, immutable_column_indexes]

                # print(var_delta)
                # print(var_delta.shape)

                masked_delta: np.array = var_delta * immutable_indicators.reshape(-1, 1)

                # print(masked_delta)
                # print(masked_delta.shape)

                delta_each_sample: np.array = masked_delta.reshape(-1)

                # print(delta_each_sample)
                # print(delta_each_sample.shape)

                assert len(delta_each_sample.shape) == 1

                is_not_changed: np.array = delta_each_sample < eps_thr

                result_array = result_array & is_not_changed

            # exit()

            result_array = result_array.astype(float)

            result_dict[f'user_immutable_{eps_thr}'] = result_array

        return result_dict

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def __init__(self, data_manager: DataCatalog, valid_indices: np.array, eps_thr_list: list):
        metric_name_list: list = [f'user_immutable_{eps_thr}' for eps_thr in eps_thr_list]
        default_dict: dict = {f'user_immutable_{eps_thr}':DEFAULT_ZERO for eps_thr in eps_thr_list}
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)
        self.eps_thr_list: list = eps_thr_list







