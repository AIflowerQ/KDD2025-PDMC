from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.catalog.catalog import DataCatalog
from predict_models.api.predict_models import MLModel


class RecourseMethod(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.

    Parameters
    ----------
    ml_model: carla.models.MLModel
        Black-box-classifier we want to discover.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Returns
    -------
    None
    """
    LOSS_FORMAT: str = '{} loss at iter {}: {}; class is {}; target is {}'

    def __init__(self, ml_model: MLModel, data_manager: DataCatalog):
        self._ml_model: MLModel = ml_model
        self._data_manager: DataCatalog = data_manager

        self.candidates_list: List[np.array] = []
        self.distances_list: List[float] = []

        self._device: torch.device = next(self._ml_model.parameters()).device

    def get_counterfactuals(self, factuals: pd.DataFrame, target_np: np.array):
        assert self._data_manager.is_transformed
        self.candidates_list = []
        self.distances_list = []

        counterfactuals: pd.DataFrame = \
            self._get_counterfactuals(factuals[self._data_manager.feature_columns_order].values, target_np)

        final_counterfactuals: pd.DataFrame = self._categorical_binary_process(counterfactuals)

        # final_counterfactuals: pd.DataFrame = counterfactuals

        print('final_counterfactuals')
        print(final_counterfactuals)

        return final_counterfactuals

    def _get_counterfactuals(self, factuals_np: np.array, target_np: np.array) -> pd.DataFrame:
        device = next(self._ml_model.parameters()).device

        factuals_tensor: torch.Tensor = torch.Tensor(factuals_np).float().to(device)
        target_tensor: torch.Tensor = torch.Tensor(target_np).float().to(device)

        list_cfs = self._counterfactual_optimization(
            factuals_tensor, target_tensor
        )

        cf_df: pd.DataFrame = pd.DataFrame(data=list_cfs, columns=self._data_manager.feature_columns_order)

        # print('cf_df')
        # print(cf_df)

        return cf_df


    def _categorical_binary_process(self, counterfactuals):
        categorical_feature_list: List[str] = self._data_manager.categorical
        counterfactual_values: np.array = counterfactuals.values
        for feature in categorical_feature_list:
            # print(feature)
            mapped_feature_indexes: List[int] = self._data_manager.locate_feature_in_encoded_ordered_list(feature)
            # print(mapped_feature_indexes)
            if len(mapped_feature_indexes) == 1:
                counterfactual_values[:, mapped_feature_indexes] = \
                    (counterfactual_values[:, mapped_feature_indexes] > 0.5).astype(float)
            else:
                max_indices = np.argmax(counterfactual_values[:, mapped_feature_indexes], axis=1)
                # print(max_indices)
                temp = np.zeros_like(counterfactual_values[:, mapped_feature_indexes])

                # 使用 numpy 的高级索引，将最大值的位置设置为 1
                temp[np.arange(counterfactual_values.shape[0]), max_indices] = 1.0
                # print(temp)

                counterfactual_values[:, mapped_feature_indexes] = temp

                # print(counterfactual_values[:, mapped_feature_indexes])
                # print(counterfactual_values[:, mapped_feature_indexes].sum(axis=1))

                if not np.allclose(counterfactual_values[:, mapped_feature_indexes].sum(axis=1), 1):
                    raise ValueError("处理后的指定列和不等于 1，请检查结果。")
                # exit()
        processed_counterfactuals: pd.DataFrame = \
            pd.DataFrame(data=counterfactual_values, columns=self._data_manager.feature_columns_order)

        # print('_categorical_binary_process:')
        # print('counterfactuals')
        # print(counterfactuals)
        # print('processed_counterfactuals')
        # print(processed_counterfactuals)
        return processed_counterfactuals

    def _counterfactual_optimization(self, factuals_tensor: torch.Tensor, target_tensor: torch.Tensor):
        # prepare data for optimization steps
        test_loader = torch.utils.data.DataLoader(
            list(zip(factuals_tensor, target_tensor)), batch_size=1, shuffle=False
        )

        list_cfs = []
        for query_instance, target_class in tqdm(test_loader):
        # for query_instance, target_class in test_loader:
            assert len(query_instance.shape) == 2
            assert len(target_class.shape) == 1

            counterfactual: np.array = self._solve_one_query(query_instance, target_class)
            list_cfs.append(counterfactual)

        return list_cfs

    def _solve_one_query(self, query_instance: torch.Tensor, target_class: torch.Tensor) -> np.array:
        self.candidates_list.clear()
        self.distances_list.clear()

        self._optimize_one_query(query_instance, target_class)

        assert len(self.candidates_list) == len(self.distances_list)

        if len(self.candidates_list) > 0:
            array_distances = np.array(self.distances_list)
            index = np.argmin(array_distances)
            result: np.array = self.candidates_list[index]
        else:
            result: np.array = np.zeros(query_instance.shape)

        result = result.reshape(-1)

        # print()
        # print(result)

        return result


    @abstractmethod
    def _optimize_one_query(self, query_instance: torch.Tensor, target_class: torch.Tensor) \
            -> (List[np.array], List[np.array]):
        pass

    def _log_candidate(self, predict: torch.Tensor, target_class: torch.Tensor,
                       candidate_np: torch.Tensor, distance: torch.Tensor):
        predicted_class: torch.Tensor = (predict > 0.5).float()
        if predicted_class == target_class:
            self.candidates_list.append(
                candidate_np.cpu().detach().numpy().reshape(-1)
            )
            self.distances_list.append(float(distance.cpu().detach().numpy()))

