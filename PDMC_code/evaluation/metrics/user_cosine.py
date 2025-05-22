import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX, DEFAULT_ZERO
from evaluation.utils import get_delta
from predict_models.api.predict_models import MLModel
from utils.utils import get_torch_model_device



def array_cos_sim(x1: np.array, x2: np.array) -> np.array:
    assert x1.shape == x2.shape
    matrix_sim = cosine_similarity(x1, x2)
    similarity = np.diag(matrix_sim)
    return similarity.reshape(-1)


class UserCosineEvaluation(EvaluationIndices):
    """
    Computes success rate for the whole recourse method.
    """

    NAME: str = 'User Cosine'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array) -> dict:

        delta: np.array = get_delta(valid_factuals, valid_counterfactuals)

        # print(delta)
        # print(delta.shape)
        # print(self.delta_gt)
        # print(self.delta_gt.shape)


        delta_cos_sim: np.ndarray = array_cos_sim(delta, self.delta_gt)
        result_dict: dict = {'user_cos': delta_cos_sim}
        # print(delta_cos_sim)

        for thr in self.eps_thr_list:

            result_array: np.ndarray = delta_cos_sim > thr
            # print(result_array)
            result_array_float = result_array.astype(float)
            result_dict[f'user_cos_{thr}'] = result_array_float

        # exit()

        return result_dict

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def __init__(self, data_manager: DataCatalog, valid_indices: np.array, eps_thr_list: list):
        metric_name_list: list = ['user_cos'] + ['user_cos_{}'.format(thr) for thr in eps_thr_list]
        default_dict: dict = {key: DEFAULT_ZERO for key in metric_name_list}
        default_dict['user_cos'] = -1.0
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)

        self.delta_gt: np.array = self.data_manager.df_delta_feature_test.values[self.valid_indices]
        self.eps_thr_list: list = eps_thr_list






