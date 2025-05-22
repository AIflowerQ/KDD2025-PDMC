import copy
import os
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from data.catalog.catalog import DataCatalog


class CategoricalProcessLayer(nn.Module):
    def __init__(self, indexes_list: List[List[int]]):

        super().__init__()
        self._indexes_list = indexes_list

    def forward(self, x: torch.Tensor):
        result: torch.Tensor = x.clone()
        for indexes in self._indexes_list:
            if len(indexes) == 1:
                result[:, indexes] = torch.sigmoid(x[:, indexes])
            else:
                softmax_output = F.softmax(x[:, indexes], dim=1)
                result[:, indexes] = softmax_output

        return result


class AbstractAutoencoder(ABC, nn.Module):
    def __init__(
            self, data_manger: DataCatalog, layers: List[int],
            mutable_mask: np.array, log_interval: int):
        super().__init__()


        self.latent_dim: int = layers[-1]
        self.data_manger: DataCatalog = data_manger
        self.layers: List[int] = layers
        self.mutable_mask: np.array = mutable_mask
        self.log_interval: int = log_interval

        self._init_encoder_decoder(layers)
        self._init_categorical_layer()

    def _init_encoder_decoder(self, layers: List[int]):
        if len(layers) < 2:
            raise ValueError(
                "Number of self._layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        assert int(np.sum(self.mutable_mask)) == layers[0]

        self._input_dim = layers[0]

        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*lst_encoder)

        # the decoder does use the immutables, so need to increase layer size accordingly.
        layers[-1] += np.sum(~self.mutable_mask)

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append((nn.ReLU()))
        self.decoder = nn.Sequential(*lst_decoder)


    def _init_categorical_layer(self):
        indexes_list: list = [
            self.data_manger.locate_feature_in_encoded_ordered_list(feature)
            for feature in self.data_manger.categorical
        ]

        self._categorical_process_layer: CategoricalProcessLayer = CategoricalProcessLayer(indexes_list)

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def fit(self, xtrain: torch.Tensor, training_para: dict):
        pass

    def validate(self, xtrain: torch.Tensor, training_para: dict):

        self.eval()
        assert len(xtrain.shape) == 2

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=training_para['batch_size']
        )
        train_loss = 0.
        train_loss_num = 0.

        criterion = nn.MSELoss(reduction="sum")
        for data in train_loader:
            reconstruction = self.predict(data)[0]
            # print(reconstruction)
            # print(data)
            loss = criterion(reconstruction, data)

            # print(reconstruction)

            train_loss += loss.item()
            train_loss_num += len(data)

        self.train()

        return train_loss / train_loss_num


class Autoencoder(AbstractAutoencoder):
    def __init__(self, data_manger: DataCatalog, layers: List[int], mutable_mask: np.array, log_interval: int = 10):
        """
        Parameters
        ----------
        data_manger:

        layers:
            List of layer sizes.
        mutable_mask:
            Mask that indicates which feature columns are mutable, and which are immutable. Setting
            all columns to mutable, results in the standard case.
        """
        super(Autoencoder, self).__init__(data_manger, layers, mutable_mask, log_interval)

        latent_dim = layers[-1]
        self._z_enc = nn.Sequential(self.encoder, nn.Linear(layers[-2], latent_dim))

        self._z_dec = nn.Sequential(
            self.decoder,
            nn.Linear(layers[1], self._input_dim),
            # nn.Sigmoid(),
        )

        self.log_str_format: str = 'AE train loss at epoch: {}, loss: {}'


    def encode(self, x):
        return self._z_enc(x)

    def decode(self, z):
        raw_decode: torch.Tensor = self._z_dec(z)
        return self._categorical_process_layer(raw_decode)

    def forward(self, x):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]
        # the mutable part gets encoded
        z = self.encode(x_mutable)
        # concatenate the immutable part to the latents and decode both
        # print(z)
        # print(x_immutable)
        z = torch.cat([z, x_immutable], dim=1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon

        return x, z

    def predict(self, data):
        self.eval()
        return self.forward(data)

    def fit(self, xtrain: torch.Tensor, training_para: dict):

        assert len(xtrain.shape) == 2

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=training_para['batch_size'], shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=training_para['lr'],
            weight_decay=training_para['lambda_reg'],
        )

        assert training_para['epochs'] > 0

        criterion = nn.MSELoss(reduction="sum")
        for epoch in range(training_para['epochs']):
            train_loss = 0
            train_loss_num = 0
            for data in train_loader:
                # forward pass
                reconstruction, _ = self(data)
                loss = criterion(reconstruction, data)
                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += len(data)

            if epoch % self.log_interval == 0:
                print(self.log_str_format.format(epoch, train_loss / train_loss_num))
                validate_str_format = self.log_str_format.replace('train', 'val')
                print(validate_str_format.format(epoch, self.validate(xtrain, training_para)))

        print(self.log_str_format.format(epoch, train_loss / train_loss_num))
        validate_str_format = self.log_str_format.replace('train', 'val')
        print(validate_str_format.format(epoch, self.validate(xtrain, training_para)))
        self.eval()


class VariationalAutoencoder(AbstractAutoencoder):
    def __init__(self, data_manger: DataCatalog, layers: List[int], mutable_mask: np.array, log_interval: int = 10):
        """
        Parameters
        ----------
        data_manger:

        layers:
            List of layer sizes.
        mutable_mask:
            Mask that indicates which feature columns are mutable, and which are immutable. Setting
            all columns to mutable, results in the standard case.
        """
        super(VariationalAutoencoder, self).__init__(data_manger, layers, mutable_mask, log_interval)

        latent_dim = layers[-1]
        self._mu_enc = nn.Sequential(self.encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(self.encoder, nn.Linear(layers[-2], latent_dim))

        self.mu_dec = nn.Sequential(
            self.decoder,
            nn.Linear(layers[1], self._input_dim),
            # nn.Sigmoid(),
        )

        self.log_str_format: str = 'VAE train loss at epoch: {}, loss: {}'


    def encode(self, x):
        return self._mu_enc(x), self._log_var_enc(x)

    def decode(self, z):
        raw_decode: torch.Tensor = self.mu_dec(z)
        return self._categorical_process_layer(raw_decode)

    @staticmethod
    def _reparametrization_trick(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]
        # the mutable part gets encoded
        mu_z, log_var_z = self.encode(x_mutable)
        z = self._reparametrization_trick(mu_z, log_var_z)
        # concatenate the immutable part to the latents and decode both
        z = torch.cat([z, x_immutable], dim=1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon

        return x, mu_z, log_var_z

    def predict(self, data):
        self.eval()
        # split up the input in a mutable and immutable part
        x = data.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]
        # the mutable part gets encoded
        mu_z, log_var_z = self.encode(x_mutable)
        z = mu_z
        # concatenate the immutable part to the latents and decode both
        z = torch.cat([z, x_immutable], dim=1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon

        # print(x)
        # print(data)

        return x, mu_z, log_var_z

    @staticmethod
    def _kld(mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

    def fit(self, xtrain: torch.Tensor, training_para: dict):

        assert len(xtrain.shape) == 2

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=training_para['batch_size'], shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=training_para['lr'],
            weight_decay=training_para['lambda_reg'],
        )

        criterion = nn.MSELoss(reduction="sum")

        # elbo = np.zeros((training_para['epochs'], 1))

        assert training_para['epochs'] > 0

        for epoch in range(training_para['epochs']):

            beta = epoch * training_para['kl_weight'] / training_para['epochs']

            train_loss = 0
            train_loss_num = 0
            for data in train_loader:
                # forward pass
                reconstruction, mu, log_var = self(data)

                recon_loss = criterion(reconstruction, data)
                kld_loss = self._kld(mu, log_var)
                loss = recon_loss + beta * kld_loss

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += len(data)

            if epoch % self.log_interval == 0:
                print(self.log_str_format.format(epoch, train_loss / train_loss_num))
                validate_str_format = self.log_str_format.replace('train', 'val')
                print(validate_str_format.format(epoch, self.validate(xtrain, training_para)))

        print(self.log_str_format.format(epoch, train_loss / train_loss_num))
        validate_str_format = self.log_str_format.replace('train', 'val')
        print(validate_str_format.format(epoch, self.validate(xtrain, training_para)))
        self.eval()


class ConditionalVariationalAutoencoder(AbstractAutoencoder):
    def __init__(
            self, data_manger: DataCatalog, layers: List[int], condition_dim: int,
            mutable_mask: np.array, log_interval: int = 10
    ):
        """
        Parameters
        ----------
        data_manger:

        layers:
            List of layer sizes.
        mutable_mask:
            Mask that indicates which feature columns are mutable, and which are immutable. Setting
            all columns to mutable, results in the standard case.
        """
        super(ConditionalVariationalAutoencoder, self).__init__(data_manger, layers, mutable_mask, log_interval)
        self.condition_dim: int = condition_dim
        self._cvae_init_encoder_decoder(layers)

        latent_dim = layers[-1]
        self._mu_enc = nn.Sequential(self.encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(self.encoder, nn.Linear(layers[-2], latent_dim))

        self._prior_mu_enc = nn.Sequential(self.prior_encoder, nn.Linear(layers[-2], latent_dim))
        self._prior_log_var_enc = nn.Sequential(self.prior_encoder, nn.Linear(layers[-2], latent_dim))

        self.mu_dec = nn.Sequential(
            self.decoder,
            nn.Linear(layers[1], layers[0]),
        )

        self.log_str_format: str = 'CVAE train loss at epoch: {}, loss: {}'


    def _cvae_init_encoder_decoder(self, layers: List[int]):
        del self.encoder
        del self.decoder

        if len(layers) < 2:
            raise ValueError(
                "Number of self._layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        assert int(np.sum(self.mutable_mask)) == layers[0]

        self._input_dim = layers[0] + self.condition_dim

        # The VAE components
        lst_encoder = []
        temp_layers: list = copy.deepcopy(layers)

        temp_layers[0] = self._input_dim
        for i in range(1, len(temp_layers) - 1):
            lst_encoder.append(nn.Linear(temp_layers[i - 1], temp_layers[i]))
            lst_encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*lst_encoder)

        lst_encoder.clear()
        temp_layers[0] = self.condition_dim
        for i in range(1, len(temp_layers) - 1):
            lst_encoder.append(nn.Linear(temp_layers[i - 1], temp_layers[i]))
            lst_encoder.append(nn.ReLU())
        self.prior_encoder = nn.Sequential(*lst_encoder)

        # the decoder does use the immutables, so need to increase layer size accordingly.

        temp_layers: list = copy.deepcopy(layers)
        temp_layers[-1] += np.sum(~self.mutable_mask)
        temp_layers[-1] += self.condition_dim
        # print(temp_layers[-1])

        lst_decoder = []
        for i in range(len(temp_layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(temp_layers[i + 1], temp_layers[i]))
            lst_decoder.append((nn.ReLU()))
        self.decoder = nn.Sequential(*lst_decoder)


    def encode(self, data):
        # [x, c]
        return self._mu_enc(data), self._log_var_enc(data)

    def prior_encode(self, condition):
        # [c]
        return self._prior_mu_enc(condition), self._prior_log_var_enc(condition)

    def decode(self, data):
        # [z, c]
        raw_decode: torch.Tensor = self.mu_dec(data)
        return self._categorical_process_layer(raw_decode)

    @staticmethod
    def _reparametrization_trick(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x, condition):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]
        # the mutable part gets encoded
        encode_input: torch.Tensor = torch.cat([x_mutable, condition], dim=1)
        mu_z, log_var_z = self.encode(encode_input)
        z = self._reparametrization_trick(mu_z, log_var_z)

        prior_mu_z, prior_log_var_z = self.prior_encode(condition)

        # concatenate the immutable part to the latents and decode both
        z = torch.cat([z, x_immutable, condition], dim=1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon

        return x, mu_z, log_var_z, prior_mu_z, prior_log_var_z

    def predict(self, data):
        self.eval()
        x, condition = self.split_x_condition(data)

        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]
        # the mutable part gets encoded
        encode_input: torch.Tensor = torch.cat([x_mutable, condition], dim=1)
        mu_z, log_var_z = self.encode(encode_input)
        z = mu_z

        prior_mu_z, prior_log_var_z = self.prior_encode(condition)

        # concatenate the immutable part to the latents and decode both
        z = torch.cat([z, x_immutable, condition], dim=1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon

        return x, mu_z, log_var_z, prior_mu_z, prior_log_var_z

    @staticmethod
    def _kld(mu, logvar, prior_mu, prior_logvar):
        temp = 1 - (prior_logvar -logvar) - \
               ((mu - prior_mu).pow(2) / prior_logvar.exp()) - (logvar.exp() / prior_logvar.exp())
        kld = -0.5 * torch.sum(temp)
        return kld

    def fit(self, xtrain: torch.Tensor, training_para: dict):

        assert len(xtrain.shape) == 2
        assert xtrain.shape[1] == (self._input_dim + int(np.sum(~self.mutable_mask)))

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=training_para['batch_size'], shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=training_para['lr'],
            weight_decay=training_para['lambda_reg'],
        )

        criterion = nn.MSELoss(reduction="sum")

        # elbo = np.zeros((training_para['epochs'], 1))

        assert training_para['epochs'] > 0

        for epoch in range(training_para['epochs']):

            beta = epoch * training_para['kl_weight'] / training_para['epochs']

            train_loss = 0
            train_loss_num = 0
            for data in train_loader:
                # forward pass
                x, condition = self.split_x_condition(data)
                reconstruction, mu, log_var, prior_mu, prior_log_var = self.forward(x, condition)

                recon_loss = criterion(reconstruction, x)
                kld_loss = self._kld(mu, log_var, prior_mu, prior_log_var)
                loss = recon_loss + beta * kld_loss

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the ways
                train_loss += loss.item()
                train_loss_num += len(data)

            if epoch % self.log_interval == 0:
                print(self.log_str_format.format(epoch, train_loss / train_loss_num))
                validate_str_format = self.log_str_format.replace('train', 'val')
                print(validate_str_format.format(epoch, self.validate(xtrain, training_para)))

        print(self.log_str_format.format(epoch, train_loss / train_loss_num))
        validate_str_format = self.log_str_format.replace('train', 'val')
        print(validate_str_format.format(epoch, self.validate(xtrain, training_para)))
        self.eval()

    def split_x_condition(self, data: torch.Tensor):

        x = data[:, :-self.condition_dim]
        condition = data[:, -self.condition_dim:]

        return x, condition

    def validate(self, xtrain: torch.Tensor, training_para: dict):

        self.eval()
        assert len(xtrain.shape) == 2

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=training_para['batch_size']
        )
        train_loss = 0.
        train_loss_num = 0.

        criterion = nn.MSELoss(reduction="sum")
        for data in train_loader:
            x, _ = self.split_x_condition(data)
            reconstruction = self.predict(data)[0]
            # print(reconstruction)
            # print(data)
            loss = criterion(reconstruction, x)

            train_loss += loss.item()
            train_loss_num += len(data)

        self.train()

        return train_loss / train_loss_num



class CSVAE(nn.Module):
    def __init__(self, data_manger: DataCatalog, layers: List[int], mutable_mask) -> None:
        super(CSVAE, self).__init__()

        self.mutable_mask = mutable_mask
        assert int(np.sum(self.mutable_mask)) == layers[0]

        self._input_dim = layers[0]
        self.z_dim = layers[-1]
        self.data_manger = data_manger
        # w_dim and labels_dim are fix due to our constraint to binary labeled data
        w_dim = 2
        self._labels_dim = w_dim

        # encoder
        lst_encoder_xy_to_w = []
        for i in range(1, len(layers) - 1):
            if i == 1:
                lst_encoder_xy_to_w.append(
                    nn.Linear(layers[i - 1] + self._labels_dim, layers[i])
                )
            else:
                lst_encoder_xy_to_w.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder_xy_to_w.append(nn.ReLU())
        self.encoder_xy_to_w = nn.Sequential(*lst_encoder_xy_to_w)

        self.mu_xy_to_w = nn.Sequential(
            self.encoder_xy_to_w, nn.Linear(layers[-2], w_dim)
        )

        self.logvar_xy_to_w = nn.Sequential(
            self.encoder_xy_to_w, nn.Linear(layers[-2], w_dim)
        )

        lst_encoder_x_to_z = copy.deepcopy(lst_encoder_xy_to_w)
        lst_encoder_x_to_z[0] = nn.Linear(layers[0], layers[1])
        self.encoder_x_to_z = nn.Sequential(*lst_encoder_x_to_z)

        self.mu_x_to_z = nn.Sequential(
            self.encoder_x_to_z, nn.Linear(layers[-2], self.z_dim)
        )

        self.logvar_x_to_z = nn.Sequential(
            self.encoder_x_to_z, nn.Linear(layers[-2], self.z_dim)
        )

        lst_encoder_y_to_w = copy.deepcopy(lst_encoder_xy_to_w)
        lst_encoder_y_to_w[0] = nn.Linear(self._labels_dim, layers[1])
        self.encoder_y_to_w = nn.Sequential(*lst_encoder_y_to_w)

        self.mu_y_to_w = nn.Sequential(
            self.encoder_y_to_w, nn.Linear(layers[-2], w_dim)
        )

        self.logvar_y_to_w = nn.Sequential(
            self.encoder_y_to_w, nn.Linear(layers[-2], w_dim)
        )

        # decoder
        # the decoder does use the immutables, so need to increase layer size accordingly.
        layers[-1] += np.sum(~mutable_mask)
        lst_decoder_zw_to_x = []
        for i in range(len(layers) - 2, 0, -1):
            if i == len(layers) - 2:
                lst_decoder_zw_to_x.append(nn.Linear(layers[i + 1] + w_dim, layers[i]))
            else:
                lst_decoder_zw_to_x.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder_zw_to_x.append(nn.ReLU())
        self.decoder_zw_to_x = nn.Sequential(*lst_decoder_zw_to_x)

        self.mu_zw_to_x = nn.Sequential(
            self.decoder_zw_to_x, nn.Linear(layers[1], self._input_dim)
        )

        self.logvar_zw_to_x = nn.Sequential(
            self.decoder_zw_to_x, nn.Linear(layers[1], self._input_dim)
        )

        lst_decoder_z_to_y = copy.deepcopy(lst_decoder_zw_to_x)
        lst_decoder_z_to_y[0] = nn.Linear(
            self.z_dim + np.sum(~mutable_mask), layers[-2]
        )
        lst_decoder_z_to_y.append(nn.Linear(layers[1], self._labels_dim))
        lst_decoder_z_to_y.append(nn.Softmax(dim=1))
        self.decoder_z_to_y = nn.Sequential(*lst_decoder_z_to_y)


        self._init_categorical_layer()

    def q_zw(self, x, y):
        xy = torch.cat([x, y], dim=1)

        z_mu = self.mu_x_to_z(x)
        z_logvar = self.logvar_x_to_z(x)

        w_mu_encoder = self.mu_xy_to_w(xy)
        w_logvar_encoder = self.logvar_xy_to_w(xy)

        w_mu_prior = self.mu_y_to_w(y)
        w_logvar_prior = self.logvar_y_to_w(y)

        return (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def p_x(self, z, w):
        zw = torch.cat([z, w], dim=1)

        mu = self.mu_zw_to_x(zw)
        logvar = self.logvar_zw_to_x(zw)

        mu = self._categorical_process_layer(mu)

        return mu, logvar

    def forward(self, x, y):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]

        (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        ) = self.q_zw(x_mutable, y)

        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        z = self.reparameterize(z_mu, z_logvar)

        # concatenate the immutable part to the latents
        z = torch.cat([z, x_immutable], dim=-1)

        zw = torch.cat([z, w_encoder], dim=1)

        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred = self.decoder_z_to_y(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = x_mu
        x_mu = x

        # set variance to zero (one in log space) for immutable features
        temp = torch.ones_like(x)
        temp[:, self.mutable_mask] = x_logvar
        x_logvar = temp

        return (
            x_mu,
            x_logvar,
            zw,
            y_pred,
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def predict(self, x, y):
        self.eval()

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, ~self.mutable_mask]

        (
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        ) = self.q_zw(x_mutable, y)

        w_encoder = w_mu_encoder
        z = z_mu

        # concatenate the immutable part to the latents
        z = torch.cat([z, x_immutable], dim=-1)

        zw = torch.cat([z, w_encoder], dim=1)

        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred = self.decoder_z_to_y(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = x_mu
        x_mu = x

        # set variance to zero (one in log space) for immutable features
        temp = torch.ones_like(x)
        temp[:, self.mutable_mask] = x_logvar
        x_logvar = temp

        return (
            x_mu,
            x_logvar,
            zw,
            y_pred,
            w_mu_encoder,
            w_logvar_encoder,
            w_mu_prior,
            w_logvar_prior,
            z_mu,
            z_logvar,
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    def fit(self, data: torch.Tensor, train_para: dict):
        epochs = train_para['epochs']
        lr = train_para['lr']
        batch_size = train_para['batch_size']

        x_train = data[:, :-2]
        y_prob_train: torch.Tensor = data[:, -2:]

        # if self._labels_dim == 2:
        #     y_prob_train: torch.Tensor = torch.zeros((data.shape[0], 2))
        #     y_prob_train[:, 0] = 1 - data[:, -1]
        #     y_prob_train[:, 1] = data[:, -1]
        # else:
        #     raise ValueError("Only binary class labels are implemented at the moment.")

        # y_prob_train.to(x_train.device)

        train_loader = torch.utils.data.DataLoader(
            list(zip(x_train, y_prob_train)), shuffle=True, batch_size=batch_size
        )

        params_without_delta = [
            param
            for name, param in self.named_parameters()
            if "decoder_z_to_y" not in name
        ]
        params_delta = [
            param for name, param in self.named_parameters() if "decoder_z_to_y" in name
        ]

        opt_without_delta = torch.optim.Adam(params_without_delta, lr=lr / 2)
        opt_delta = torch.optim.Adam(params_delta, lr=lr / 2)



        for i in range(epochs):
            train_x_recon_losses = []
            train_y_recon_losses = []
            for x, y in train_loader:
                (
                    loss_val,
                    x_recon_loss_val,
                    w_kl_loss_val,
                    z_kl_loss_val,
                    y_negentropy_loss_val,
                    y_recon_loss_val,
                ) = csvae_loss(self, x, y, train_para)

                opt_delta.zero_grad()
                y_recon_loss_val.backward(retain_graph=True)

                opt_without_delta.zero_grad()
                loss_val.backward()

                opt_without_delta.step()
                opt_delta.step()

                train_x_recon_losses.append(x_recon_loss_val.item())
                train_y_recon_losses.append(y_recon_loss_val.item())

            print(f'epoch {i}: x recon {np.mean(train_x_recon_losses)}, y recon {np.mean(train_y_recon_losses)}')

        self.eval()

    def _init_categorical_layer(self):
        indexes_list: list = [
            self.data_manger.locate_feature_in_encoded_ordered_list(feature)
            for feature in self.data_manger.categorical
        ]

        self._categorical_process_layer: CategoricalProcessLayer = CategoricalProcessLayer(indexes_list)


def csvae_loss(csvae, x_train, y_train, train_para: dict):
    x = x_train.clone().float()
    y = y_train.clone().float()

    (
        x_mu,
        x_logvar,
        zw,
        y_pred,
        w_mu_encoder,
        w_logvar_encoder,
        w_mu_prior,
        w_logvar_prior,
        z_mu,
        z_logvar,
    ) = csvae.forward(x, y)

    x_recon = nn.MSELoss()(x_mu, x)

    w_dist = dists.MultivariateNormal(
        w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp())
    )
    w_prior = dists.MultivariateNormal(
        w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp())
    )
    w_kl = dists.kl.kl_divergence(w_dist, w_prior)

    z_dist = dists.MultivariateNormal(
        z_mu.flatten(), torch.diag(z_logvar.flatten().exp())
    )
    z_prior = dists.MultivariateNormal(
        torch.zeros(csvae.z_dim * z_mu.size()[0], device=z_mu.device),
        torch.eye(csvae.z_dim * z_mu.size()[0], device=z_mu.device),
    )
    z_kl = dists.kl.kl_divergence(z_dist, z_prior)

    y_pred_negentropy = (
        y_pred.log() * y_pred + (1 - y_pred).log() * (1 - y_pred)
    ).mean()

    class_label = torch.argmax(y, dim=1)
    y_recon = (
        train_para['y_recon_weight']
        * torch.where(
            class_label == 1, -torch.log(y_pred[:, 1]), -torch.log(y_pred[:, 0])
        )
    ).mean()

    ELBO = x_recon + train_para['z_kl_weight'] * z_kl + train_para['w_kl_weight'] * w_kl \
           + train_para['y_pred_negentropy_weight'] * y_pred_negentropy

    return ELBO, x_recon, w_kl, z_kl, y_pred_negentropy, y_recon


