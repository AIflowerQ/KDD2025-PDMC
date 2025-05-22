from typing import Dict, List

import numpy as np
import torch
from torch import nn

from data.catalog.catalog import DataCatalog
from predict_models.api.predict_models import MLModel
from recourse_methods.api.recourse_method import RecourseMethod
from PDMC_code.recourse_methods.processing import merge_default_parameters


class SingleCF(RecourseMethod):

    _DEFAULT_HYPERPARAMS = {
        "lambda": 0.5,
        "lr": 0.1,
        "max_iter": 1000,
        'log_interval': 10,
        'silent': False
    }
    NAME: str = 'SingleCF'

    def __init__(
            self, ml_model: MLModel,
            data_manager: DataCatalog,
            hyperparams: Dict = None
    ) -> None:
        super().__init__(ml_model, data_manager)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._target_column = data_manager.target
        self._lambda = self._params["lambda"]
        self._lr = self._params["lr"]
        self._max_iter = self._params["max_iter"]

    def _optimize_one_query(self, query_instance: torch.Tensor, target_class: torch.Tensor) \
            -> (List[np.array], List[np.array]):
        cf = query_instance.clone().detach().requires_grad_(True)
        assert len(cf.shape) == 2

        optim = torch.optim.Adam([cf], self._lr)

        for idx in range(self._max_iter):

            output = self._ml_model(cf)
            assert len(output.shape) == 1

            cf.requires_grad = True

            loss, sup_loss = self._compute_loss(cf, query_instance, target_class, output)

            loss.backward()
            optim.step()
            optim.zero_grad()
            cf.detach_()

            if idx % self._params['log_interval'] == 0 and not self._params['silent']:
                print(self.LOSS_FORMAT.format(
                    self.NAME, idx,
                    loss.detach().cpu().numpy(),
                    float(output.detach().cpu().numpy()),
                    float(target_class.detach().cpu().numpy())
                ))
            self._log_candidate(output, target_class, cf, sup_loss)

    def _compute_loss(self, cf_initialize, query_instance, target, predict):

        loss_function = nn.BCELoss()

        # classification loss
        loss1 = loss_function(predict, target)
        # distance loss
        loss2 = torch.norm((cf_initialize - query_instance), dim=1, p=2)

        return loss1 + self._lambda * loss2, self._lambda * loss2

