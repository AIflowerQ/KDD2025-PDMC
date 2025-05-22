import json
from copy import copy

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from predict_models.train.dataset import ElasticDataSet
from predict_models.train.evaluate import BasicEvaluator
import torch.nn.functional as F

from utils.utils import merge_dict, sum_and_mean_merge_dict_func, transfer_loss_dict_to_line_str

OPTIMIZER_MAP: dict = {
    'adam': Adam,
    # 'sgd': SGD
}

LOSS_FUNC_MAP: dict = {
    'BCE': F.binary_cross_entropy
}

class BasicTrainer:
    def __init__(self, model: torch.nn.Module, evaluator: BasicEvaluator,
                 device: torch.device,
                 training_data: ElasticDataSet, batch_size: int,
                 epochs: int, evaluate_interval: int, lr: float,
                 weight_decay: float, test_begin_epoch: int = 0,
                 silent: bool = False, shuffle: bool = False, all_data_on_device: bool = True,
                 need_pre_test: bool = True
                 ):

        self.model: torch.nn.Module = model
        self.evaluator: BasicEvaluator = evaluator
        self.device: torch.device = device
        self.training_data: ElasticDataSet = training_data
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.evaluate_interval: int = evaluate_interval
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.test_begin_epoch: int = test_begin_epoch
        self.all_data_on_device: bool = all_data_on_device
        self.need_pre_test: bool = need_pre_test

        self.optimizer_name: str = 'adam'

        self.optimizer_class = OPTIMIZER_MAP[self.optimizer_name]

        self.loss_func_name = 'BCE'

        self.loss_func = LOSS_FUNC_MAP[self.loss_func_name]

        self.silent: bool = silent

        self.shuffle: bool = shuffle

        # self.data_iter: list = [batch_data for batch_data in mini_batch(self.batch_size, *(self.training_data[:]))]

        self.optimizer: torch.optim.Optimizer = \
            self.optimizer_class(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # print(self._optimizer)

        self.epoch_cnt = 0

        self.test_result_list = []
        self.test_epoch_list = []

        self.train_loss_list = []
        self.train_epoch_list = []

    def train_a_batch(self, batch_data: list) -> dict:
        x_tensor: torch.Tensor = batch_data[0] if self.all_data_on_device else batch_data[0].to(self.device)
        y_tensor: torch.Tensor = batch_data[1] if self.all_data_on_device else batch_data[1].to(self.device)
        y_tensor = y_tensor.float()

        # print(x_tensor)
        # print(y_tensor)

        model_pred = self.model(x_tensor)
        model_pred: torch.Tensor
        # print(model_pred)

        loss: torch.Tensor = self.loss_func(model_pred, y_tensor, reduction="mean")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(5, time.time())

        loss_dict: dict = {
            'loss': float(loss)
        }

        return loss_dict

    def train_an_epoch(self) -> dict:

        assert self.epoch_cnt < self.epochs

        self.model.train()

        loss_dicts_list: list = list()

        data_loader: DataLoader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=self.shuffle)

        for batch_data in data_loader:
            loss_dict = self.train_a_batch(batch_data)
            loss_dicts_list.append(loss_dict)

        if len(loss_dicts_list) < 1:
            loss_dicts_list.append({'loss': 0.0})

        mean_loss_dict: dict = \
            merge_dict(loss_dicts_list, sum_and_mean_merge_dict_func, total_num=len(self.training_data))

        if not self.silent:
            print('train epoch: {} ['.format(self.epoch_cnt), transfer_loss_dict_to_line_str(mean_loss_dict), ']')

        self.epoch_cnt += 1

        return mean_loss_dict

    def _log_train_loss_in_list(self, loss_dict):
        self.train_loss_list.append(loss_dict)
        self.train_epoch_list.append(self.epoch_cnt)

    def _log_test_result_in_list(self, result_dict):
        self.test_result_list.append(result_dict)
        self.test_epoch_list.append(self.epoch_cnt)

    def train(self):

        if not self.silent:
            print('Begin Training~')

        if self.need_pre_test and self.evaluator is not None:
            no_train_test_result: dict = self.evaluator.evaluate()
            # input()
            self._log_test_result_in_list(no_train_test_result)

            if not self.silent:
                print('test result at epoch: {}'.format(self.epoch_cnt))
                print(json.dumps(no_train_test_result, indent=4))

        while self.epoch_cnt < self.epochs:
            epoch_loss_dict: dict = self.train_an_epoch()
            self._log_train_loss_in_list(epoch_loss_dict)

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch \
                    and self.evaluator is not None:
                # evaluator control whether print test result
                test_result_dict: dict = self.evaluator.evaluate()
                self._log_test_result_in_list(test_result_dict)

                if not self.silent:
                    print('test result at epoch: {}'.format(self.epoch_cnt))
                    print(json.dumps(test_result_dict, indent=4))

        return self.log_lists

    @property
    def log_lists(self):
        return (copy(self.train_loss_list), copy(self.train_epoch_list)), \
               (copy(self.test_result_list), copy(self.test_epoch_list))


