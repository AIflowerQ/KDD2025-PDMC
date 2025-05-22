import numpy as np
import pandas as pd
import torch

from causal_structure.api.causal_structure import CausalStructure, CausalStructureAllKnow
from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX, DEFAULT_MIN
from predict_models.api.predict_models import MLModel
from utils.utils import get_torch_model_device


class CausalConstraintEvaluation(EvaluationIndices):
    """
    Computes success rate for the whole recourse method.
    """
    NAME: str = 'Causal Constraint'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array):

        endogenous_feature_indices_map: dict = self.causal_structure.endogenous_feature_indices_map

        # print(endogenous_feature_indices_map)

        endogenous_features_list: list = list(endogenous_feature_indices_map.keys())

        factual_pdf_values_dict: dict = \
            self.causal_structure.endogenous_variable_probability_density(valid_factuals, endogenous_features_list)

        # exit()

        counterfactual_pdf_values_dict: dict = \
            self.causal_structure.endogenous_variable_probability_density(valid_counterfactuals, endogenous_features_list)

        ratios_list: list = []

        for var in endogenous_feature_indices_map.keys():
            factual_pdf: np.array = factual_pdf_values_dict[var]
            counterfactual_pdf: np.array = counterfactual_pdf_values_dict[var]

            # print()
            # print(counterfactual_pdf)
            # print(factual_pdf)

            ratio = (counterfactual_pdf / (factual_pdf + 1e-6)).reshape(-1)

            ratios_list.append(ratio)

        # print(ratios_list)
        stacked_arrays = np.stack(ratios_list)
        # print(stacked_arrays)
        ratios_var_mean = np.mean(stacked_arrays, axis=0)
        # print(ratios_var_mean)

        assert ratios_var_mean.shape[0] == valid_factuals.shape[0]

        result_dict: dict = {
            'pdf_ratio': ratios_var_mean
        }

        for thr in self.thr_list:
            thr_result: np.array = (ratios_var_mean > thr).astype(float)

            result_dict['pdf_thr_{}'.format(thr)] = thr_result

        return result_dict

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def __init__(
            self, data_manager: DataCatalog, valid_indices: np.array,
            causal_structure: CausalStructureAllKnow, thr_list: list
    ):
        # print('chipichipi')
        metric_name_list: list = ['pdf_ratio'] + ['pdf_thr_{}'.format(thr) for thr in thr_list]
        default_dict: dict = {key: DEFAULT_MIN for key in metric_name_list}

        self.thr_list = thr_list
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)

        self.causal_structure: CausalStructureAllKnow = causal_structure
        # print(type(self.causal_structure))





