import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX, DEFAULT_ZERO
from evaluation.utils import get_delta
from predict_models.api.predict_models import MLModel
from utils.utils import get_torch_model_device


class GlobalMonotonicEvaluation(EvaluationIndices):
    """
    Computes success rate for the whole recourse method.
    """

    NAME: str = 'Global Monotonic'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array) -> dict:

        assert self.data_manager.is_transformed

        delta: np.array = get_delta(valid_factuals, valid_counterfactuals)

        result_array: np.array = np.full(valid_factuals.shape[0], True, dtype=bool)

        increase_list, decrease_list = self.data_manager.monotonic_variables

        result_dict: dict = dict()

        for eps_thr in self.eps_thr_list:

            # print(increase_list)
            # print(decrease_list)

            for increase_var in increase_list:
                # print(increase_var)
                immutable_column_index: int = self.data_manager.locate_feature_in_encoded_ordered_list(increase_var)[0]

                var_delta: np.array = delta[:, immutable_column_index]

                not_decrease: np.array = var_delta > (-eps_thr)

                # print(result_array)
                # print(not_decrease)

                result_array = result_array & not_decrease

            for decrease_var in decrease_list:

                immutable_column_index: int = self.data_manager.locate_feature_in_encoded_ordered_list(decrease_var)[0]

                var_delta: np.array = delta[:, immutable_column_index]

                not_increase: np.array = var_delta < eps_thr

                result_array = result_array & not_increase

            result_array_float = result_array.astype(float)
            result_dict[f'global_monotonic_{eps_thr}'] = result_array_float

        return result_dict

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def __init__(self, data_manager: DataCatalog, valid_indices: np.array, eps_thr_list: list):
        metric_name_list: list = [f'global_monotonic_{eps_thr}' for eps_thr in eps_thr_list]
        default_dict: dict = {f'global_monotonic_{eps_thr}':DEFAULT_ZERO for eps_thr in eps_thr_list}
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)
        self.eps_thr_list: list = eps_thr_list







