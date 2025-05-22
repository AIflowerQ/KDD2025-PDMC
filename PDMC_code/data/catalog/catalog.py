from abc import ABC
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.base import BaseEstimator

from data.pipelining import (
    decode,
    descale,
    encode,
    fit_encoder,
    fit_scaler,
    scale,
)

from ..api import Data


class DataCatalog(Data, ABC):
    def __init__(
        self,
        data_name: str,
        df,
        df_train,
        df_train_new,
        df_test,
        df_test_new,
        scaling_method: str = "Standard",
        encoding_method: str = "OneHot_drop_binary",
    ):
        self.name = data_name
        # df is used to estimate many para
        # df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0)
        self._df = df
        self._df_train = df_train
        self._df_train_new = df_train_new
        self._df_test = df_test
        self._df_test_new = df_test_new

        # Fit scaler and encoder
        self.scaler: BaseEstimator = fit_scaler(
            scaling_method, self.df[self.continuous]
        )
        self.encoder: BaseEstimator = fit_encoder(
            encoding_method, self.df[self.categorical]
        )
        self._identity_encoding = (
            encoding_method is None or encoding_method == "Identity"
        )

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data

        self._is_transformed: bool = False

        self.transform_all()

        self._df_train = self.df_train.sort_values(by=self.id, ascending=True)
        self._df_train_new = self.df_train_new.sort_values(by=self.id, ascending=True)
        self._df_test = self.df_test.sort_values(by=self.id, ascending=True)
        self._df_test_new = self.df_test_new.sort_values(by=self.id, ascending=True)

        assert self.df_train[self.id].equals(self.df_train_new[self.id])
        assert self.df_test[self.id].equals(self.df_test_new[self.id])

    @property
    def is_transformed(self) -> bool:
        return self._is_transformed

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_train_new(self) -> pd.DataFrame:
        return self._df_train_new.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()

    @property
    def df_test_new(self) -> pd.DataFrame:
        return self._df_test_new.copy()

    @property
    def df_delta_feature_train(self) -> pd.DataFrame:
        return self.df_train_new[self.feature_columns_order] - self.df_train[self.feature_columns_order]

    @property
    def df_delta_feature_test(self) -> pd.DataFrame:
        return self.df_test_new[self.feature_columns_order] - self.df_test[self.feature_columns_order]

    @property
    def scaler(self) -> BaseEstimator:
        """
        Contains a fitted sklearn scaler.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler: BaseEstimator):
        """
        Sets a new fitted sklearn scaler.

        Parameters
        ----------
        scaler : sklearn.preprocessing.Scaler
            Fitted scaler for ML model.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        self._scaler = scaler

    @property
    def encoder(self) -> BaseEstimator:
        """
        Contains a fitted sklearn encoder:

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: BaseEstimator):
        """
        Sets a new fitted sklearn encoder.

        Parameters
        ----------
        encoder: sklearn.preprocessing.Encoder
            Fitted encoder for ML model.
        """
        self._encoder = encoder

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to keep correct encodings and normalization

        Parameters
        ----------
        df : pd.DataFrame
            Contains raw (not normalized and not encoded) data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input normalized and encoded

        """
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def transform_all(self):

        assert self.is_transformed is False

        self._df = self.transform(self.df)
        self._df_train = self.transform(self.df_train)
        self._df_train_new = self.transform(self.df_train_new)
        self._df_test = self.transform(self.df_test)
        self._df_test_new = self.transform(self.df_test_new)
        self._is_transformed = True

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms output after prediction back into original form.
        Only possible for DataFrames with preprocessing steps.

        Parameters
        ----------
        df : pd.DataFrame
            Contains normalized and encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction output denormalized and decoded

        """
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def inverse_transform_all(self):

        assert self.is_transformed

        self._df = self.inverse_transform(self.df)
        self._df_train = self.inverse_transform(self.df_train)
        self._df_train_new = self.inverse_transform(self.df_train_new)
        self._df_test = self.inverse_transform(self.df_test)
        self._df_test_new = self.inverse_transform(self.df_test_new)
        self._is_transformed = False

    def get_pipeline_element(self, key: str) -> Callable:
        """
        Returns a specific element of the transformation pipeline.

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("encoder", lambda x: decode(self.encoder, self.categorical, x)),
            ("scaler", lambda x: descale(self.scaler, self.continuous, x)),
        ]

    @property
    def feature_columns_order(self) -> List[str]:
        if self.is_transformed:
            # print(1)
            if len(self.categorical) == 0:
                # print(11)
                return self.continuous
            else:
                # print(12)
                encoded_features: list = self.categorical_transformed
                return self.continuous + encoded_features
        else:
            # print(2)
            return self.continuous + self.categorical

    @property
    def categorical_transformed(self):
        if len(self.categorical) != 0:
            return self.encoder.get_feature_names(self.categorical).tolist()
        else:
            return []

    def locate_feature_in_encoded_ordered_list(self, feature: str) -> List[int]:

        assert self.is_transformed

        if len(self.categorical) == 0:
            raise ValueError

        # if feature not in self.categorical:
        #     raise ValueError

        column_indexes: list = []

        for idx, encoded_feature in enumerate(self.feature_columns_order):

            if feature not in encoded_feature:
                continue

            column_indexes.append(idx)

        assert len(column_indexes) > 0

        return column_indexes

    @property
    def continuous_scalar_para(self):
        assert self.is_transformed
        if isinstance(self.scaler, preprocessing.MinMaxScaler):
            stats_dict = {feature: {'min': _min, 'max': _max} for feature, _min, _max in
                      zip(self.continuous, self.scaler.data_min_, self.scaler.data_max_)}

        elif isinstance(self.scaler, preprocessing.StandardScaler):
            stats_dict = {feature: {'mean': mean, 'std': std} for feature, mean, std in
                      zip(self.continuous, self.scaler.mean_, self.scaler.scale_)}

        else:
            raise ValueError

        return stats_dict


    @property
    def user_immutables_individual_truth_test(self) -> dict:
        raise NotImplementedError

    @property
    def monotonic_variables(self):
        raise NotImplementedError

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        raise NotImplementedError

