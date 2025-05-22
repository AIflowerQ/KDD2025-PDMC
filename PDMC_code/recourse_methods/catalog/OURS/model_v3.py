from typing import Dict, List

import numpy as np
import torch
from sklearn.covariance import MinCovDet
from torch import nn

from data.catalog.catalog import DataCatalog
from predict_models.api.predict_models import MLModel
from recourse_methods.api.recourse_method import RecourseMethod
from recourse_methods.autoencoder.autoencoder import ConditionalVariationalAutoencoder
from PDMC_code.recourse_methods.processing import merge_default_parameters
from utils.utils import mahalanobis_distance_torch


class OURS_V3(RecourseMethod):

    _DEFAULT_HYPERPARAMS = {
        "md_epsilon": 1e-5,
        "l2_coe": 0.5,
        "delta_x_l2_coe": 0.5,
        "delta_x_given_x_l2_coe": 0.5,
        "delta_x_md_coe": 0.5,
        "delta_x_given_x_md_coe": 0.5,
        "x_md_coe": 0.5,
        "lr": 0.1,
        "max_iter": 1000,
        'samples_num': 2000,
        "log_interval": 10,
        "silent": True

    }
    NAME: str = 'OURS_V3'

    def __init__(
            self, ml_model: MLModel,
            data_manager: DataCatalog,
            p_x_given_y_cvae: ConditionalVariationalAutoencoder,
            p_delta_x_given_x_cvae: ConditionalVariationalAutoencoder,
            hyperparams: Dict = None
    ) -> None:
        super().__init__(ml_model, data_manager)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._target_column = data_manager.target

        self._md_epsilon: float = self._params["md_epsilon"]
        self.l2_coe = self._params["l2_coe"]
        self.delta_x_l2_coe = self._params["delta_x_l2_coe"]
        self.delta_x_given_x_l2_coe = self._params["delta_x_given_x_l2_coe"]
        self.delta_x_md_coe = self._params["delta_x_md_coe"]
        self.delta_x_given_x_md_coe = self._params["delta_x_given_x_md_coe"]
        self.x_md_coe = self._params["x_md_coe"]

        self._lr = self._params["lr"]
        self._max_iter = self._params["max_iter"]
        self._samples_num: int = self._params["samples_num"]

        self.p_x_given_y_cvae: ConditionalVariationalAutoencoder = p_x_given_y_cvae.eval()
        self.p_delta_x_given_x_cvae: ConditionalVariationalAutoencoder = p_delta_x_given_x_cvae.eval()

        assert np.sum(p_x_given_y_cvae.mutable_mask) == p_x_given_y_cvae.mutable_mask.shape[0]
        assert np.sum(p_delta_x_given_x_cvae.mutable_mask) == p_delta_x_given_x_cvae.mutable_mask.shape[0]

        self.y_0_mean_tensor, self.y_0_cov_inv_tensor, self.y_1_mean_tensor, self.y_1_cov_inv_tensor \
            = self._prepare_x_md_cov_inv()

        self.delta_x_mean_tensor, self.delta_x_cov_inv_tensor = self._prepare_delta_x_md_cov_inv()

        self.delta_x_given_x_mean_tensor, self.delta_x_given_x_cov_inv_tensor = None, None

        self.p_delta_x_samples: torch.Tensor = \
            torch.tensor(self._data_manager.df_delta_feature_train.values, device=self._device)

        self.debug_helper = MinCovDet(random_state=17373331)
        self.debug_helper.fit(self._data_manager.df_delta_feature_train.values)

        print('debug_helper')
        print(self._data_manager.df_delta_feature_train.values)

    def _prepare_delta_x_md_cov_inv(self):
        training_delta_x: np.array = self._data_manager.df_delta_feature_train.values
        training_delta_x_mean = np.mean(training_delta_x, axis=0)

        cov_matrix: np.ndarray = np.cov(training_delta_x, rowvar=False)  # 计算协方差矩阵
        cov_matrix = cov_matrix + self._md_epsilon * np.eye(cov_matrix.shape[0])
        cov_matrix: np.ndarray = cov_matrix

        cov_inv: np.ndarray = np.linalg.inv(cov_matrix)

        delta_x_mean_tensor: torch.Tensor = torch.Tensor(training_delta_x_mean).to(self._device)
        delta_x_cov_inv_tensor: torch.Tensor = torch.Tensor(cov_inv).to(self._device)

        return delta_x_mean_tensor, delta_x_cov_inv_tensor

    def _prepare_delta_x_given_x_md_cov_inv(self, samples: torch.Tensor):
        samples_np: np.ndarray = samples.detach().cpu().numpy()
        training_mean = np.mean(samples_np, axis=0)

        cov_matrix: np.ndarray = np.cov(samples_np, rowvar=False)  # 计算协方差矩阵
        cov_matrix = cov_matrix + self._md_epsilon * np.eye(cov_matrix.shape[0])
        cov_matrix: np.ndarray = cov_matrix

        cov_inv: np.ndarray = np.linalg.inv(cov_matrix)

        mean_tensor: torch.Tensor = torch.Tensor(training_mean).to(self._device)
        cov_inv_tensor: torch.Tensor = torch.Tensor(cov_inv).to(self._device)

        return mean_tensor, cov_inv_tensor

    def _prepare_x_md_cov_inv(self):
        training_x: np.array = self._data_manager.df[self._data_manager.feature_columns_order].values
        training_y: np.array = self._data_manager.df[self._data_manager.target].to_numpy()
        y_0_indicator: np.array = (training_y == 0)
        y_1_indicator: np.array = (training_y == 1)
        # print(y_0_indicator)
        # print(y_1_indicator)

        assert np.sum(y_1_indicator) + np.sum(y_0_indicator) == len(training_y)

        y_0_training_x: np.array = training_x[y_0_indicator]
        y_1_training_x: np.array = training_x[y_1_indicator]

        assert len(y_0_training_x) + len(y_1_training_x) == len(training_x)

        y_0_training_delta_x_mean = np.mean(y_0_training_x, axis=0)
        y_0_cov_matrix: np.ndarray = np.cov(y_0_training_x, rowvar=False)  # 计算协方差矩阵
        y_0_cov_matrix = y_0_cov_matrix + self._md_epsilon * np.eye(y_0_cov_matrix.shape[0])
        y_0_cov_matrix: np.ndarray = y_0_cov_matrix
        y_0_cov_inv: np.ndarray = np.linalg.inv(y_0_cov_matrix)

        y_1_training_delta_x_mean = np.mean(y_1_training_x, axis=0)
        y_1_cov_matrix: np.ndarray = np.cov(y_1_training_x, rowvar=False)  # 计算协方差矩阵
        y_1_cov_matrix = y_1_cov_matrix + self._md_epsilon * np.eye(y_1_cov_matrix.shape[0])
        y_1_cov_matrix: np.ndarray = y_1_cov_matrix
        y_1_cov_inv: np.ndarray = np.linalg.inv(y_1_cov_matrix)

        y_0_mean_tensor: torch.Tensor = torch.Tensor(y_0_training_delta_x_mean).to(self._device)
        y_0_cov_inv_tensor: torch.Tensor = torch.Tensor(y_0_cov_inv).to(self._device)

        y_1_mean_tensor: torch.Tensor = torch.Tensor(y_1_training_delta_x_mean).to(self._device)
        y_1_cov_inv_tensor: torch.Tensor = torch.Tensor(y_1_cov_inv).to(self._device)

        return y_0_mean_tensor, y_0_cov_inv_tensor, y_1_mean_tensor, y_1_cov_inv_tensor


    def _generate_samples(self, condition: torch.Tensor):

        condition = condition.reshape(1, self.p_delta_x_given_x_cvae.condition_dim)
        prior_mu, prior_log_var = self.p_delta_x_given_x_cvae.prior_encode(condition)

        prior_mu = prior_mu.reshape(1, self.p_delta_x_given_x_cvae.latent_dim)
        prior_log_var = prior_log_var.reshape(1, self.p_delta_x_given_x_cvae.latent_dim)

        raw_latent_samples: torch.Tensor = \
            torch.randn([self._samples_num, self.p_delta_x_given_x_cvae.latent_dim], device=self._device)

        prior_std: torch.Tensor = torch.exp(0.5 * prior_log_var)

        condition_latent_samples: torch.Tensor = prior_mu + raw_latent_samples * prior_std

        condition_repeat: torch.Tensor = condition.repeat(self._samples_num, 1)

        condition_decode_input: torch.Tensor = torch.cat([condition_latent_samples, condition_repeat], dim=1)
        p_delta_x_given_x_samples: torch.Tensor = self.p_delta_x_given_x_cvae.decode(condition_decode_input)

        return p_delta_x_given_x_samples.detach()


    def _optimize_one_query(self, query_instance: torch.Tensor, target_class: torch.Tensor) \
            -> (List[np.array], List[np.array]):

        y_pred: torch.Tensor = self._ml_model.predict(query_instance).reshape(-1, 1)
        y_pred_one_hot: torch.Tensor = torch.zeros((query_instance.shape[0], 2)).to(query_instance.device)
        y_pred_one_hot[:, 0] = 1 - y_pred
        y_pred_one_hot[:, 1] = y_pred

        target_class_one_hot: torch.Tensor = torch.zeros_like(y_pred_one_hot)
        target_class_one_hot[:, 0] = 1 - target_class
        target_class_one_hot[:, 1] = target_class

        query_instance_cat: torch.Tensor = torch.cat([query_instance, y_pred_one_hot], dim=1)

        z = self.p_x_given_y_cvae.encode(query_instance_cat)[0]
        z = z.clone().detach().requires_grad_(True)

        optim = torch.optim.Adam([z], self._lr)


        # cf = query_instance.clone().detach().requires_grad_(True)
        # optim = torch.optim.Adam([cf], self._lr)

        p_delta_x_samples = self.p_delta_x_samples

        p_delta_x_given_x_samples = self._generate_samples(query_instance)

        self.delta_x_given_x_mean_tensor, self.delta_x_given_x_cov_inv_tensor = \
            self._prepare_delta_x_given_x_md_cov_inv(p_delta_x_given_x_samples)

        # print('mahalanobis target')
        # print(p_delta_x_samples[0, :].reshape(1, -1))
        # print(self.debug_helper.mahalanobis(p_delta_x_samples[0, :].reshape(1, -1).detach().cpu().numpy()))

        for idx in range(self._max_iter):
            z_y_cat: torch.Tensor = torch.cat([z, target_class_one_hot], dim=1)
            cf = self.p_x_given_y_cvae.decode(z_y_cat)

            assert len(cf.shape) == 2

            output = self._ml_model(cf)
            assert len(output.shape) == 1

            z.requires_grad = True

            loss, sup_loss = \
                self._compute_loss(
                    cf, query_instance, target_class, output,
                    p_delta_x_samples, p_delta_x_given_x_samples
                )

            loss.backward()
            optim.step()
            optim.zero_grad()
            # cf.detach_()

            if idx % self._params['log_interval'] == 0 and not self._params['silent']:
                print(self.LOSS_FORMAT.format(
                    self.NAME, idx,
                    loss.detach().cpu().numpy(),
                    float(output.detach().cpu().numpy()),
                    float(target_class.detach().cpu().numpy())
                ))

            self._log_candidate(output, target_class, cf, sup_loss)
            # print('mahalanobis')
            # print(self.debug_helper.mahalanobis(cf.detach().cpu().numpy()))
        # print('end')

    def _compute_loss(self, cf_initialize, query_instance, target, predict, p_delta_x_samples, p_delta_x_given_x_samples):

        loss_function = nn.BCELoss()

        # classification loss
        target_loss = loss_function(predict, target)
        # distance loss
        proximity_loss = self.l2_coe * torch.norm((cf_initialize - query_instance), dim=1, p=2)


        delta_x: torch.Tensor = cf_initialize - query_instance
        delta_x_l2: torch.Tensor = \
            self._samples_based_l2_distance(delta_x, p_delta_x_samples)
        delta_x_given_x_l2: torch.Tensor = \
            self._samples_based_l2_distance(delta_x, p_delta_x_given_x_samples)

        delta_x_l2_loss = self.delta_x_l2_coe * delta_x_l2
        delta_x_given_x_l2_loss = self.delta_x_given_x_l2_coe * delta_x_given_x_l2

        delta_x_md: torch.Tensor = self._delta_x_samples_based_mahalanobis_distance(delta_x)
        delta_x_given_x_md: torch.Tensor = \
            self._delta_x_given_x_samples_based_mahalanobis_distance(delta_x)

        x_md: torch.Tensor = self._x_given_y_samples_based_mahalanobis_distance(
            cf_initialize, target
        )

        delta_x_md_loss = delta_x_md * self.delta_x_md_coe

        delta_x_given_x_md_loss = delta_x_given_x_md * self.delta_x_given_x_md_coe

        x_md_loss = x_md * self.x_md_coe

        sup_loss: torch.Tensor = proximity_loss + delta_x_l2_loss + delta_x_given_x_l2_loss \
                                 + delta_x_md_loss + delta_x_given_x_md_loss + x_md_loss

        # temp_dict: dict = {
        #     'delta_x_l2': float(delta_x_l2.detach().cpu().numpy()),
        #     'delta_x_given_x_l2': float(delta_x_given_x_l2.detach().cpu().numpy()),
        #     'delta_x_md': float(delta_x_md.detach().cpu().numpy()),
        #     'delta_x_given_x_md': float(delta_x_given_x_md.detach().cpu().numpy()),
        #     'x_md': float(x_md.detach().cpu().numpy()),
        # }
        #
        # print(json.dumps(temp_dict, indent=4))

        loss: torch.Tensor = target_loss + sup_loss

        return loss, sup_loss

    def _delta_x_samples_based_mahalanobis_distance(self, instance: torch.Tensor):
        return mahalanobis_distance_torch(instance, self.delta_x_mean_tensor, self.delta_x_cov_inv_tensor)

    def _x_given_y_samples_based_mahalanobis_distance(self, instance: torch.Tensor, target_class: torch.Tensor):
        y_0_md: torch.Tensor = mahalanobis_distance_torch(instance, self.y_0_mean_tensor, self.y_0_cov_inv_tensor)
        y_1_md: torch.Tensor = mahalanobis_distance_torch(instance, self.y_1_mean_tensor, self.y_1_cov_inv_tensor)

        md: torch.Tensor = (y_0_md * (1 - target_class)) + (y_1_md * target_class)

        return md

    def _delta_x_given_x_samples_based_mahalanobis_distance(
            self, instance: torch.Tensor
    ):
        return mahalanobis_distance_torch(
            instance, self.delta_x_given_x_mean_tensor, self.delta_x_given_x_cov_inv_tensor
        )

    @staticmethod
    def _samples_based_l2_distance(instance: torch.Tensor, samples: torch.Tensor):
        distances: torch.Tensor = instance - samples
        assert distances.shape == samples.shape
        distance = torch.mean(torch.norm(distances, dim=1, p=2))

        return distance






