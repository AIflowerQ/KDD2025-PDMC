import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from torch.utils.data import DataLoader

from predict_models.api.predict_models import MLModel
from predict_models.train.dataset import ElasticDataSet


class BasicEvaluator:
    def __init__(
            self, model: MLModel, device: torch.device,
            test_dataset: ElasticDataSet,
            test_batch_size: int,
            all_data_on_device: bool = True
    ):

        self.model: MLModel = model
        self.device = device

        self.test_x_tensor, self.test_y_tensor = test_dataset[:]
        self.test_x_tensor: torch.Tensor
        self.test_y_tensor: torch.Tensor
        self.test_dataset: ElasticDataSet = test_dataset

        if all_data_on_device:
            self.test_x_tensor = self.test_x_tensor.to(device)
            self.test_y_tensor = self.test_y_tensor.to(device)

            self.test_dataset = ElasticDataSet(self.test_x_tensor, self.test_y_tensor)

        self.test_dataloader: DataLoader = DataLoader(self.test_dataset, batch_size=test_batch_size)

        assert len(self.test_x_tensor.shape) == 2
        assert len(self.test_y_tensor.shape) == 1

        self.test_batch_size: int = test_batch_size
        self.all_data_on_device: bool = all_data_on_device

    def predict_batch_tabular_data(self, input_tensor: torch.Tensor) -> torch.Tensor:

        if not self.all_data_on_device:
            input_tensor = input_tensor.to(self.device)

        # print(input_tensor)

        predict_tensor: torch.Tensor = self.model.predict(input_tensor)

        return predict_tensor

    def test(self) -> dict:
        self.model.eval()

        pred_y_array_list: list = []

        for batch_data in self.test_dataloader:
            pred_y: torch.Tensor = self.predict_batch_tabular_data(batch_data[0])

            pred_y_numpy: np.array = pred_y.cpu().numpy().reshape(-1)

            pred_y_array_list.append(pred_y_numpy)

        pred_y_numpy_cat: np.array = np.concatenate(pred_y_array_list, axis=0)
        binary_pred_y_numpy_cat: np.array = np.zeros_like(pred_y_numpy_cat)
        binary_pred_y_numpy_cat[pred_y_numpy_cat > 0.5] = 1.0
        binary_pred_y_numpy_cat = binary_pred_y_numpy_cat.astype(int)

        true_y_numpy: np.array = self.test_dataset[:][1].cpu().numpy().astype(int).reshape(-1)

        cut_pred_y_numpy_cat: np.array = np.copy(pred_y_numpy_cat)
        cut_pred_y_numpy_cat[cut_pred_y_numpy_cat >= 1.0] = 1.0 - 1e-7
        cut_pred_y_numpy_cat[cut_pred_y_numpy_cat <= 0.0] = 0.0 + 1e-7

        auc: float = float(roc_auc_score(true_y_numpy, pred_y_numpy_cat))
        bce: float = float(log_loss(true_y_numpy, cut_pred_y_numpy_cat))
        acc: float = float(accuracy_score(true_y_numpy, binary_pred_y_numpy_cat))

        result_dict: dict = {
            'auc': auc,
            'bce': bce,
            'acc': acc
        }

        return result_dict

    def evaluate(self, *args) -> dict:

        return self.test()
