from typing import List

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor, KernelDensity

from data.catalog.catalog import DataCatalog
from evaluation.api.evaluation import EvaluationIndices, DEFAULT_MAX, DEFAULT_ZERO
from evaluation.utils import get_delta
from utils.utils import mahalanobis_distance


class DeltaXManifoldEvaluation(EvaluationIndices):

    NAME: str = 'Delta X Manifold'

    def valid_process(self, valid_factuals: np.array, valid_counterfactuals: np.array) -> dict:
        result: dict = dict()

        delta: np.array = get_delta(valid_factuals, valid_counterfactuals)

        result.update(self.calculate_md(delta))
        result.update(self.calculate_kde(delta))
        # result.update(self.calculate_lof(delta))

        return result

    def invalid_process(self, invalid_factuals: np.array, invalid_counterfactuals: np.array):
        raise NotImplementedError

    def calculate_md(self, x: np.array):
        # return {'md': self.mcd.mahalanobis(x)}
        # pass

        # print(mahalanobis_distance(x, self.training_delta_x_mean, self.cov_inv))
        # exit()

        return {'md': mahalanobis_distance(x, self.training_delta_x_mean, self.cov_inv)}


    def calculate_kde(self, x: np.array):
        log_dens = self.kde.score_samples(x)
        dens = np.exp(log_dens)
        # print(dens)

        percentile = np.sum(self.pdf_train <= dens[:, None], axis=1) / len(self.pdf_train) # * 100

        # print(percentile)
        # exit()

        return {'kde_pdf': dens, 'kde_percentile': percentile}

    # def calculate_lof(self, x: np.array):
    #     result: dict = dict()
    #     for k in self.lof_dict.keys():
    #         result['lof_{}'.format(k)] = -self.lof_dict[k].decision_function(x)
    #         # print(k)
    #         # print(self.lof_dict[k].decision_function(x))
    #
    #     return result

    def __init__(
            self, data_manager: DataCatalog, valid_indices: np.array,
            md_epsilon: float, kde_bw: float,
    ):
        metric_name_list: list = ['md', 'kde_pdf', 'kde_percentile']
        default_dict: dict = {
            'md': DEFAULT_MAX,
            'kde_pdf': DEFAULT_ZERO,
            'kde_percentile': DEFAULT_ZERO
        }
        super().__init__(data_manager, valid_indices, metric_name_list, default_dict, False)

        training_delta_x: np.array = self.data_manager.df_delta_feature_train.values

        # print('training_delta_x')
        # print(training_delta_x)
        # self.mcd = MinCovDet(random_state=random_state).fit(training_delta_x)

        self.md_epsilon: float = md_epsilon

        self.training_delta_x_mean = np.mean(training_delta_x, axis=0)

        cov_matrix: np.ndarray = np.cov(training_delta_x, rowvar=False)  # 计算协方差矩阵
        cov_matrix = cov_matrix + self.md_epsilon * np.eye(cov_matrix.shape[0])
        self.cov_matrix: np.ndarray = cov_matrix

        # print(self.cov_matrix)
        # print(np.linalg.eigvals(self.cov_matrix))
        self.cov_inv: np.ndarray = np.linalg.inv(self.cov_matrix)
        # print(np.linalg.eigvals(self.cov_inv))

        # exit()

        self.kde: KernelDensity = KernelDensity(bandwidth=kde_bw)

        self.kde.fit(training_delta_x)

        log_dens_train = self.kde.score_samples(training_delta_x)
        self.pdf_train = np.exp(log_dens_train)

        # exit()
        #
        # self.lof_dict: dict = {
        #     k: LocalOutlierFactor(n_neighbors=k, novelty=True).fit(training_delta_x) for k in k_list
        # }










