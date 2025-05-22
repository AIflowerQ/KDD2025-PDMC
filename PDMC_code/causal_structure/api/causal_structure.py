from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch


class CausalStructure(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def causal_reconstruct(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        :param x: 2-d
        :return: 2-d
        exogenous variable will keep unchanged
        endogenous variable will be assigned based on causal relationship.

        """

        pass



class CausalStructureAllKnow(CausalStructure, ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def endogenous_feature_indices_map(self) -> dict:
        """
        not y, str -> int
        :return:
        """
        pass

    @abstractmethod
    def endogenous_variable_probability_density(self, x: np.array, variables_list: List[str]) -> dict:
        """
        :param x: 2-d
        :param variables_list: must be an endogenous variable
        :return: dict of (1-d probability density of each sample)
        """
        pass





