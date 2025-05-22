from typing import Union

import numpy as np
import torch

from data.catalog.catalog import DataCatalog
from recourse_methods.autoencoder.autoencoder import Autoencoder, VariationalAutoencoder, \
    ConditionalVariationalAutoencoder, CSVAE


def prepare_autoencoder_basic(
        ae: Union[Autoencoder, VariationalAutoencoder],
        data_manager: DataCatalog, training_para: dict,
        device: torch.device
):
    train_np: np.array = data_manager.df[data_manager.feature_columns_order].values
    train_tensor: torch.Tensor = torch.Tensor(train_np).to(device).float()

    ae.fit(train_tensor, training_para)


def prepare_autoencoder_xy_train(
        ae: Union[ConditionalVariationalAutoencoder, CSVAE],
        data_manager: DataCatalog, training_para: dict,
        device: torch.device
):
    train_x_np: np.array = data_manager.df[data_manager.feature_columns_order].values
    train_x_tensor: torch.Tensor = torch.Tensor(train_x_np).to(device).float()

    train_y_np: np.array = data_manager.df[data_manager.target].to_numpy().reshape(-1)
    train_y_one_hot_np: np.array = np.eye(2)[train_y_np.astype(int)]
    # print(train_y_one_hot_np)
    # exit()
    train_y_one_hot_tensor: torch.Tensor = torch.Tensor(train_y_one_hot_np).to(device).reshape(-1, 2).float()

    #
    # print(train_x_tensor)
    # print(train_y_one_hot_tensor)

    train_data: torch.Tensor = torch.cat([train_x_tensor, train_y_one_hot_tensor], dim=1)
    # print(train_data)
    #
    # exit()

    ae.fit(train_data, training_para)


def prepare_autoencoder_delta_x_train(
        ae: VariationalAutoencoder,
        data_manager: DataCatalog, training_para: dict,
        device: torch.device
):
    train_np: np.array = data_manager.df_delta_feature_train[data_manager.feature_columns_order].values
    train_tensor: torch.Tensor = torch.Tensor(train_np).to(device).float()

    # print(train_tensor)

    ae.fit(train_tensor, training_para)


def prepare_autoencoder_delta_x_given_x_train(
        ae: ConditionalVariationalAutoencoder,
        data_manager: DataCatalog, training_para: dict,
        device: torch.device
):
    train_delta_x_np: np.array = data_manager.df_delta_feature_train[data_manager.feature_columns_order].values
    train_delta_x_tensor: torch.Tensor = torch.Tensor(train_delta_x_np).to(device).float()

    train_x_np: np.array = data_manager.df_train[data_manager.feature_columns_order].values
    train_x_tensor: torch.Tensor = torch.Tensor(train_x_np).to(device).float()

    # print(train_delta_x_tensor)
    # print(train_x_tensor)

    train_data_tensor: torch.Tensor = torch.cat([train_delta_x_tensor, train_x_tensor], dim=1)

    # print(train_data_tensor)

    ae.fit(train_data_tensor, training_para)

