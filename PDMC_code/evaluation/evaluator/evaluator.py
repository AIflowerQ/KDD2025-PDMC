from typing import List, Union

import numpy as np
import pandas as pd
import torch

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices
from predict_models.api.predict_models import MLModel
from recourse_methods.api.recourse_method import RecourseMethod


class Evaluator:
    def __init__(
            self, data_manager: DataCatalog, ml_model: MLModel,
            recourse_method: RecourseMethod,
            metric_type_para_dict: dict,
            qualified_metric_dict: dict,
            target: Union[str, int, bool],
            device: torch.device
    ):
        self.data_manager: DataCatalog = data_manager
        self.ml_model: MLModel = ml_model
        self.recourse_method: RecourseMethod = recourse_method
        self.metric_type_para_dict: dict = metric_type_para_dict
        self.qualified_metric_dict: dict = qualified_metric_dict

        self.target: Union[str, int, bool] = target

        self.device: torch.device = device

        predict_label: np.ndarray = self._predict_label(self.data_manager.df_test)
        if isinstance(target, str):
            assert target == 'inverse'
            self.target_labels: np.ndarray = 1 - predict_label

        else:
            self.target_labels: np.ndarray = np.zeros_like(predict_label)

            self.target_labels += self.target



    def _predict_label(self, x_df: pd.DataFrame) -> np.array:

        x_tensor: torch.Tensor = torch.tensor(
            x_df[self.data_manager.feature_columns_order].values
        ).float().to(self.device)

        predict_label: np.ndarray = self.ml_model.predict(x_tensor).cpu().numpy()

        predict_label[predict_label > 0.5] = 1.0
        predict_label[predict_label <= 0.5] = 0.0

        predict_label =predict_label.astype(int)

        return predict_label

    def evaluate(self):

        counterfactuals: pd.DataFrame = \
            self.recourse_method.get_counterfactuals(self.data_manager.df_test, self.target_labels)

        predict_labels: np.ndarray = self._predict_label(counterfactuals)

        valid_indices: np.ndarray = predict_labels == self.target_labels

        valid_rate: float = float(np.sum(valid_indices) / len(predict_labels))

        evaluate_mean_result: dict = {'valid_rate': valid_rate}

        evaluate_result: dict = dict()

        for metric_type in self.metric_type_para_dict.keys():
            para: dict = self.metric_type_para_dict[metric_type]
            if 'target_labels' in para.keys():
                para['target_labels'] = self.target_labels

            print(metric_type.NAME)
            metric: EvaluationIndices = metric_type(data_manager=self.data_manager, valid_indices=valid_indices, **para)

            metric_mean_result: dict
            metric_result: dict
            metric_mean_result, metric_result = \
                metric.get_evaluation(
                    factuals=self.data_manager.df_test[self.data_manager.feature_columns_order].values,
                    counterfactuals=counterfactuals[self.data_manager.feature_columns_order].values
                )

            evaluate_mean_result[metric.NAME] = metric_mean_result
            evaluate_result[metric.NAME] = metric_result
            # evaluate_mean_result.update(metric_mean_result)

        qualified_rate: float = self._qualified_metric_calculate(int(np.sum(valid_indices)), evaluate_result)

        evaluate_mean_result['qualified_rate'] = qualified_rate

        return evaluate_mean_result


    def _qualified_metric_calculate(self, arr_len: int, raw_evaluate_result: dict) -> float:

        result_array: np.array = np.full(arr_len, True, dtype=bool)

        for evaluation in self.qualified_metric_dict.keys():
            # print(evaluation.NAME)
            metric: str = self.qualified_metric_dict[evaluation]

            raw_metric_result: np.ndarray = raw_evaluate_result[evaluation.NAME][metric]
            # print(raw_metric_result)

            assert arr_len == raw_metric_result.shape[0]

            raw_metric_result = raw_metric_result.astype(np.bool)

            result_array = result_array & raw_metric_result
        #     print(result_array)
        #     print()
        #
        # print(result_array)

        return float(np.mean(result_array.astype(float)))
