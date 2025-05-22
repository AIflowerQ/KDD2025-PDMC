from typing import List, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from causal_structure.api.causal_structure import CausalStructureAllKnow
from data.catalog.datasat_catalog import BnPlusCatalog


class BnPlusStandardProcessCausalModelStandard(CausalStructureAllKnow):

    def __init__(self, data_manager: BnPlusCatalog, k1 = 0.0003, b1 = 10.0, sigma3 = 0.5):
        super().__init__()

        self.data_manager: BnPlusCatalog = data_manager

        self.k1: float = k1
        self.b1: float = b1
        self.sigma3: float = sigma3

        self.endogenous_features_list: list = ['x3']


        self.x1_index, self.x1_mu_est, self.x1_sigma_est = self._get_estimated_standard_paras('x1')
        self.x2_index, self.x2_mu_est, self.x2_sigma_est = self._get_estimated_standard_paras('x2')
        self.x3_index, self.x3_mu_est, self.x3_sigma_est = self._get_estimated_standard_paras('x3')

        self.x3_new_sigma: float = self.sigma3 / self.x3_sigma_est

    def gen_x3_no_noise(self, x1: np.array, x2: np.array) -> np.array:
        assert self.data_manager.is_transformed

        new_square: np.array = (self.x1_sigma_est * x1 + self.x2_sigma_est * x2 + self.x1_mu_est + self.x2_mu_est) ** 2

        return (self.k1 * new_square + self.b1 - self.x3_mu_est) / self.x3_sigma_est

    def gen_x3_no_noise_tensor(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert self.data_manager.is_transformed

        new_square: torch.Tensor = (self.x1_sigma_est * x1 + self.x2_sigma_est * x2 + self.x1_mu_est + self.x2_mu_est) ** 2

        return (self.k1 * new_square + self.b1 - self.x3_mu_est) / self.x3_sigma_est

    @property
    def endogenous_feature_indices_map(self) -> dict:
        # print(23333)
        return {'x3': self.x3_index}

    def _get_estimated_standard_paras(self, feature: str):
        assert feature in ['x1', 'x2', 'x3']
        assert self.data_manager.is_transformed
        # print(feature)
        index: int = self.data_manager.locate_feature_in_encoded_ordered_list(feature)[0]

        paras_dict: dict = self.data_manager.continuous_scalar_para[feature]

        est_mu: float = paras_dict['mean']
        est_sigma: float = paras_dict['std']

        return index, est_mu, est_sigma

    def endogenous_variable_probability_density(self, x: np.array, variables_list: List[str]) -> np.array:
        # print('endogenous_variable_probability_density')
        # print(x)
        # print(x.shape)


        assert variables_list[0] == 'x3'
        assert len(variables_list) == 1

        # print('x')
        # print(x)

        x_new: np.array = self.causal_reconstruct(x)

        # print('x_new')
        # print(x_new)

        x3_observed: np.array = x[:, self.x3_index].reshape(-1)
        x3_mu: np.array = x_new[:, self.x3_index].reshape(-1)

        # print('x3_observed')
        # print(x3_observed)
        #
        # print('x3_mu')
        # print(x3_mu)

        pdf_values: np.array = stats.norm.pdf(x3_observed, loc=x3_mu, scale=self.x3_new_sigma)

        return {'x3': pdf_values}


    def causal_reconstruct_np(self, x: np.array) -> np.array:
        x1: np.array = x[:, self.x1_index].reshape(-1)
        x2: np.array = x[:, self.x2_index].reshape(-1)
        # print('x1:')
        # print(x1)
        # print('x2:')
        # print(x2)
        reconstruct_x3: np.array = self.gen_x3_no_noise(x1, x2)

        # print('raw_x3:')
        # print(x[:, self.x3_index])
        # print('reconstruct_x3:')
        # print(reconstruct_x3)
        # print()

        x_new: np.array = np.copy(x)

        x_new[:, self.x3_index] = reconstruct_x3

        return x_new

    def causal_reconstruct_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = x[:, self.x1_index].reshape(-1)
        x2: torch.Tensor = x[:, self.x2_index].reshape(-1)

        reconstruct_x3: torch.Tensor = self.gen_x3_no_noise_tensor(x1, x2)

        x_new: torch.Tensor = torch.clone(x)

        x_new[:, self.x3_index] = reconstruct_x3

        return x_new

    def causal_reconstruct(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # print(x)
        # print(type(x))
        if isinstance(x, np.ndarray):
            return self.causal_reconstruct_np(x)
        elif isinstance(x, torch.Tensor):
            return self.causal_reconstruct_tensor(x)
        else:
            raise NotImplementedError




class BnPlusStandardProcessCausalModel(CausalStructureAllKnow):

    def __init__(self, data_manager: BnPlusCatalog, k1 = 0.0003, b1 = 10.0, sigma3 = 0.5):
        super().__init__()

        self.data_manager: BnPlusCatalog = data_manager

        self.k1: float = k1
        self.b1: float = b1
        self.sigma3: float = sigma3

        self.endogenous_features_list: list = ['x3']

        self.x1_index: int = self.data_manager.locate_feature_in_encoded_ordered_list('x1')[0]
        self.x2_index: int = self.data_manager.locate_feature_in_encoded_ordered_list('x2')[0]
        self.x3_index: int = self.data_manager.locate_feature_in_encoded_ordered_list('x3')[0]

    def gen_x3_no_noise(self, x1: np.array, x2: np.array) -> np.array:
        assert self.data_manager.is_transformed

        return (x1 + x2)**2 * self.k1 + self.b1

    def gen_x3_no_noise_tensor(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert self.data_manager.is_transformed

        return (x1 + x2) ** 2 *self.k1 + self.b1

    @property
    def endogenous_feature_indices_map(self) -> dict:
        # print(23333)
        return {'x3': self.x3_index}

    def endogenous_variable_probability_density(self, x: np.array, variables_list: List[str]) -> np.array:

        # print('endogenous_variable_probability_density')
        # print(x)
        # print(x.shape)

        x_df: pd.DataFrame = pd.DataFrame(data=x, columns=self.data_manager.feature_columns_order)

        # print(x_df)

        x_inverse_df: pd.DataFrame = self.data_manager.inverse_transform(x_df)

        # print(x_inverse_df)

        assert variables_list[0] == 'x3'
        assert len(variables_list) == 1

        # print('x')
        # print(x)

        x_inverse_np: np.ndarray = x_inverse_df.values

        x_new: np.array = self.causal_reconstruct(x_inverse_np)

        # print('x_new')
        # print(x_new)

        x3_observed: np.array = x_inverse_np[:, self.x3_index].reshape(-1)
        x3_mu: np.array = x_new[:, self.x3_index].reshape(-1)

        # print('x3_observed')
        # print(x3_observed)
        #
        # print('x3_mu')
        # print(x3_mu)

        pdf_values: np.array = stats.norm.pdf(x3_observed, loc=x3_mu, scale=self.sigma3)

        return {'x3': pdf_values}


    def causal_reconstruct_np(self, x: np.array) -> np.array:
        x1: np.array = x[:, self.x1_index].reshape(-1)
        x2: np.array = x[:, self.x2_index].reshape(-1)
        # print('x1:')
        # print(x1)
        # print('x2:')
        # print(x2)
        reconstruct_x3: np.array = self.gen_x3_no_noise(x1, x2)
        #
        # print('raw_x3:')
        # print(x[:, self.x3_index])
        # print('reconstruct_x3:')
        # print(reconstruct_x3)
        # print()

        x_new: np.array = np.copy(x)

        x_new[:, self.x3_index] = reconstruct_x3

        return x_new

    def causal_reconstruct_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = x[:, self.x1_index].reshape(-1)
        x2: torch.Tensor = x[:, self.x2_index].reshape(-1)

        reconstruct_x3: torch.Tensor = self.gen_x3_no_noise_tensor(x1, x2)

        x_new: torch.Tensor = torch.clone(x)

        x_new[:, self.x3_index] = reconstruct_x3

        return x_new

    def causal_reconstruct(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # print(x)
        # print(type(x))
        if isinstance(x, np.ndarray):
            return self.causal_reconstruct_np(x)
        elif isinstance(x, torch.Tensor):
            return self.causal_reconstruct_tensor(x)
        else:
            raise NotImplementedError



