from typing import List

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor, KernelDensity

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX, DEFAULT_ZERO
from evaluation.utils import get_delta
from utils.utils import mahalanobis_distance


class XManifoldEvaluation(EvaluationIndices):

    NAME: str = 'X Manifold'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array) -> dict:
        result: dict = dict()

        target_labels: np.array = self.target_labels[self.valid_indices]

        result.update(self.calculate_md(valid_counterfactuals, target_labels))
        result.update(self.calculate_kde(valid_counterfactuals, target_labels))
        # result.update(self.calculate_lof(valid_counterfactuals, target_labels))

        return result

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def calculate_md(self, x: np.array, target_labels: np.array):
        assert target_labels.shape[0] == x.shape[0]
        #
        # md_y_0: np.array = self.mcd_y_0.mahalanobis(x) * (target_labels == 0).astype(float)
        # md_y_1: np.array = self.mcd_y_1.mahalanobis(x) * (target_labels == 1).astype(float)

        md_y_0: np.ndarray = \
            mahalanobis_distance(
                x, self.y_0_training_delta_x_mean, self.y_0_cov_inv
            ) * (target_labels == 0).astype(float)
        md_y_1: np.ndarray = \
            mahalanobis_distance(
                x, self.y_1_training_delta_x_mean, self.y_1_cov_inv
            ) * (target_labels == 1).astype(float)

        md: np.array = md_y_1 + md_y_0

        # print(md_y_0)
        # print(md_y_1)
        # print(md)
        #
        # exit()

        return {'md': md}


    def calculate_kde(self, x: np.array, target_labels: np.array):
        y_0_log_dens = self.y_0_kde.score_samples(x)
        y_0_dens = np.exp(y_0_log_dens) * (target_labels == 0).astype(float)

        y_1_log_dens = self.y_1_kde.score_samples(x)
        y_1_dens = np.exp(y_1_log_dens) * (target_labels == 1).astype(float)

        dens: np.ndarray = y_0_dens + y_1_dens

        percentile = np.sum(self.pdf_train <= dens[:, None], axis=1) / len(self.pdf_train) # * 100

        # print(y_0_dens)
        # print(y_1_dens)
        # print(dens)
        #
        # print(percentile)
        # exit()

        return {'kde_pdf': dens, 'kde_percentile': percentile}



    # def calculate_lof(self, x: np.array, target_labels: np.array):
    #     assert target_labels.shape[0] == x.shape[0]
    #     result: dict = dict()
    #     for k in self.lof_dict_y_0.keys():
    #         lof_y_0: np.array = -self.lof_dict_y_0[k].decision_function(x) * (target_labels == 0).astype(float)
    #         lof_y_1: np.array = -self.lof_dict_y_1[k].decision_function(x) * (target_labels == 1).astype(float)
    #         result['lof_{}'.format(k)] = lof_y_0 + lof_y_1
    #     return result

    def __init__(
            self, data_manager: DataCatalog, valid_indices: np.array,
            md_epsilon: float,
            kde_bw: float,
            target_labels: np.array
    ):
        metric_name_list: list = ['md', 'kde_pdf', 'kde_percentile']
        default_dict: dict = {
            'md': DEFAULT_MAX,
            'kde_pdf': DEFAULT_ZERO,
            'kde_percentile': DEFAULT_ZERO
        }
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)

        self.target_labels: np.array = target_labels

        training_x: np.array = self.data_manager.df[self.data_manager.feature_columns_order].values
        training_y: np.array = self.data_manager.df[self.data_manager.target].to_numpy()
        y_0_indicator: np.array = (training_y == 0)
        y_1_indicator: np.array = (training_y == 1)
        # print(y_0_indicator)
        # print(y_1_indicator)

        assert np.sum(y_1_indicator) + np.sum(y_0_indicator) == len(training_y)

        y_0_training_x: np.array = training_x[y_0_indicator]
        y_1_training_x: np.array = training_x[y_1_indicator]

        assert len(y_0_training_x) + len(y_1_training_x) == len(training_x)

        self.md_epsilon: float = md_epsilon

        self.y_0_training_delta_x_mean = np.mean(y_0_training_x, axis=0)
        y_0_cov_matrix: np.ndarray = np.cov(y_0_training_x, rowvar=False)  # 计算协方差矩阵
        y_0_cov_matrix = y_0_cov_matrix + self.md_epsilon * np.eye(y_0_cov_matrix.shape[0])
        self.y_0_cov_matrix: np.ndarray = y_0_cov_matrix
        self.y_0_cov_inv: np.ndarray = np.linalg.inv(self.y_0_cov_matrix)

        self.y_1_training_delta_x_mean = np.mean(y_1_training_x, axis=0)
        y_1_cov_matrix: np.ndarray = np.cov(y_1_training_x, rowvar=False)  # 计算协方差矩阵
        y_1_cov_matrix = y_1_cov_matrix + self.md_epsilon * np.eye(y_1_cov_matrix.shape[0])
        self.y_1_cov_matrix: np.ndarray = y_1_cov_matrix
        self.y_1_cov_inv: np.ndarray = np.linalg.inv(self.y_1_cov_matrix)



        self.y_0_kde: KernelDensity = KernelDensity(bandwidth=kde_bw)
        self.y_0_kde.fit(y_0_training_x)
        y_0_log_dens_train = self.y_0_kde.score_samples(training_x)
        self.y_0_pdf_train = np.exp(y_0_log_dens_train) * (training_y == 0).astype(float)

        self.y_1_kde: KernelDensity = KernelDensity(bandwidth=kde_bw)
        self.y_1_kde.fit(y_1_training_x)
        y_1_log_dens_train = self.y_1_kde.score_samples(training_x)
        self.y_1_pdf_train = np.exp(y_1_log_dens_train) * (training_y == 1).astype(float)

        self.pdf_train: np.ndarray = self.y_0_pdf_train + self.y_1_pdf_train

        # print(self.y_0_pdf_train)
        # print(self.y_1_pdf_train)
        # print(self.pdf_train)
        # exit()


        # self.mcd_y_0 = MinCovDet(random_state=random_state).fit(y_0_training_x)
        # self.mcd_y_1 = MinCovDet(random_state=random_state).fit(y_1_training_x)

        # self.lof_dict_y_0: dict = {
        #     k: LocalOutlierFactor(n_neighbors=k, novelty=True).fit(y_0_training_x) for k in k_list
        # }
        # self.lof_dict_y_1: dict = {
        #     k: LocalOutlierFactor(n_neighbors=k, novelty=True).fit(y_1_training_x) for k in k_list
        # }











