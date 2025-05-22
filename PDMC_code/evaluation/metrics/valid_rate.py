import numpy as np
import pandas as pd
import torch

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import Evaluation
from predict_models.api.predict_models import MLModel
from utils.utils import get_torch_model_device


class ValidRate(Evaluation):
    """
    Computes success rate for the whole recourse method.
    """

    def __init__(self, data_manager: DataCatalog, ml_model: MLModel, target_labels: np.array):
        raise NotImplementedError
        super().__init__(data_manager)
        self.metric_name: str = "valid_rate"
        self.ml_model: MLModel = ml_model

        target_labels = target_labels.astype(int)

        self.target_labels: np.array = target_labels

        assert np.all((target_labels == 0) | (target_labels == 1))
        assert len(target_labels.shape) == 1

        self._valid_indices: np.array = None

    def get_evaluation(self, factuals, counterfactuals) -> dict:
        counterfactuals_tensor: torch.Tensor = \
            torch.from_numpy(counterfactuals).to(get_torch_model_device(self.ml_model))

        predict_labels: np.array = self.ml_model.predict(counterfactuals_tensor).reshape(-1).cpu().numpy()

        assert np.all((predict_labels >= 0) & (predict_labels <= 1))

        predict_labels = np.where(predict_labels > 0.5, 1, 0).astype(int)

        assert predict_labels.shape == self.target_labels.shape

        valid_indices: np.array = np.where(predict_labels == self.target_labels)[0]
        valid_cnt: int = len(valid_indices)

        valid_rate: float = valid_cnt / len(predict_labels)

        return {self.metric_name: valid_rate}

    @property
    def valid_indices(self) -> np.array:
        return self._valid_indices






