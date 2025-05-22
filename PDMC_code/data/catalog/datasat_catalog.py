from typing import List, Union

import numpy as np
import pandas as pd
import torch

from .catalog import DataCatalog


class BnPlusCatalog(DataCatalog):
    @property
    def categorical(self):
        return ['x7']

    @property
    def continuous(self):
        return ['x{}'.format(i) for i in range(1, 7)]

    @property
    def immutables(self):
        return ['x5', 'x7']

    @property
    def user_immutables(self):
        return ['x6']

    @property
    def monotonic_variables(self):
        return ['x4'], []

    @property
    def target(self):
        return 'y'

    @property
    def id(self):
        return 'id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        assert self.is_transformed
        x6_is_immutable: np.array = (self.df_test['x7_1'] == 0).values.reshape(-1)
        return {'x6': x6_is_immutable}

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        assert feature == 'x6'
        assert len(cfs.shape) == 2

        x7_index = self.locate_feature_in_encoded_ordered_list('x7')

        x6_is_immutable = (cfs[:, x7_index] == 0).reshape(-1)

        return x6_is_immutable

    def __init__(self, file_path: str):

        data_name: str = 'BN_Plus'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')[:100]
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')[:100]

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)



class BnPlusCatalogData1(DataCatalog):
    @property
    def categorical(self):
        return ['x7']

    @property
    def continuous(self):
        return ['x{}'.format(i) for i in range(1, 7)]

    @property
    def immutables(self):
        return ['x5', 'x7']

    @property
    def user_immutables(self):
        return ['x6']

    @property
    def monotonic_variables(self):
        return [], ['x4']

    @property
    def target(self):
        return 'y'

    @property
    def id(self):
        return 'id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        assert self.is_transformed
        x6_is_immutable: np.array = (self.df_test['x7_1'] == 0).values.reshape(-1)
        return {'x6': x6_is_immutable}

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        assert feature == 'x6'
        assert len(cfs.shape) == 2

        x7_index = self.locate_feature_in_encoded_ordered_list('x7')

        x6_is_immutable = (cfs[:, x7_index] == 0).reshape(-1)

        return x6_is_immutable

    def __init__(self, file_path: str):

        data_name: str = 'BN_Plus'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')[:100]
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')[:100]

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)


class BnPlusCatalogData2(DataCatalog):
    @property
    def categorical(self):
        return ['x7']

    @property
    def continuous(self):
        return ['x{}'.format(i) for i in range(1, 7)]

    @property
    def immutables(self):
        return ['x5', 'x7']

    @property
    def user_immutables(self):
        return ['x6']

    @property
    def monotonic_variables(self):
        return ['x4'], []

    @property
    def target(self):
        return 'y'

    @property
    def id(self):
        return 'id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        assert self.is_transformed
        x6_is_immutable: np.array = (self.df_test['x7_1'] == 0).values.reshape(-1)
        return {'x6': x6_is_immutable}

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        assert feature == 'x6'
        assert len(cfs.shape) == 2

        x7_index = self.locate_feature_in_encoded_ordered_list('x7')

        x6_is_immutable = (cfs[:, x7_index] == 0).reshape(-1)

        return x6_is_immutable

    def __init__(self, file_path: str):

        data_name: str = 'BN_Plus'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')[:100]
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')[:100]

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)



class BnPlusCatalogData3(DataCatalog):
    @property
    def categorical(self):
        return ['x7']

    @property
    def continuous(self):
        return ['x{}'.format(i) for i in range(1, 7)]

    @property
    def immutables(self):
        return ['x5', 'x7']

    @property
    def user_immutables(self):
        return ['x6']

    @property
    def monotonic_variables(self):
        return [], ['x4']

    @property
    def target(self):
        return 'y'

    @property
    def id(self):
        return 'id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        assert self.is_transformed
        x6_is_immutable: np.array = (self.df_test['x7_1'] == 0).values.reshape(-1)
        return {'x6': x6_is_immutable}

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        assert feature == 'x6'
        assert len(cfs.shape) == 2

        x7_index = self.locate_feature_in_encoded_ordered_list('x7')

        x6_is_immutable = (cfs[:, x7_index] == 0).reshape(-1)

        return x6_is_immutable

    def __init__(self, file_path: str):

        data_name: str = 'BN_Plus'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')[:100]
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')[:100]

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)



class BnPlusCatalogData4(DataCatalog):
    @property
    def categorical(self):
        return ['x7']

    @property
    def continuous(self):
        return ['x{}'.format(i) for i in range(1, 7)]

    @property
    def immutables(self):
        return ['x5', 'x7']

    @property
    def user_immutables(self):
        return ['x6']

    @property
    def monotonic_variables(self):
        return ['x4'], []

    @property
    def target(self):
        return 'y'

    @property
    def id(self):
        return 'id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        assert self.is_transformed
        x6_is_immutable: np.array = (self.df_test['x7_1'] == 0).values.reshape(-1)
        return {'x6': x6_is_immutable}

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        assert feature == 'x6'
        assert len(cfs.shape) == 2

        x7_index = self.locate_feature_in_encoded_ordered_list('x7')

        x6_is_immutable = (cfs[:, x7_index] == 0).reshape(-1)

        return x6_is_immutable

    def __init__(self, file_path: str):

        data_name: str = 'BN_Plus'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')[:100]
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')[:100]

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)


class MimicPressureFeatureSelect(DataCatalog):
    @property
    def categorical(self):
        return []

    @property
    def continuous(self):
        return [
            'heart rate',
            'red blood cell count',
            'sodium',
            'mean blood pressure',
            'systemic vascular resistance',
            'glucose',
            'chloride urine',
            'glascow coma scale total',
            'hematocrit',
            'positive end-expiratory pressure set',
            'respiratory rate',
            'prothrombin time pt',
            'cholesterol',
            'hemoglobin',
            'creatinine',
            'blood urea nitrogen',
            'bicarbonate',
            'calcium ionized',
            'partial pressure of carbon dioxide',
            'magnesium',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
            'calcium urine'
        ]

    @property
    def immutables(self):
        return []

    @property
    def user_immutables(self):
        return []

    @property
    def monotonic_variables(self):
        return [], []

    @property
    def target(self):
        return 'diastolic blood pressure Y'

    @property
    def id(self):
        return 'subject_id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        raise NotImplementedError

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        raise NotImplementedError

    def __init__(self, file_path: str):

        data_name: str = 'MIMIC_pressure_feature_select'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)


class MimicPressure(DataCatalog):
    @property
    def categorical(self):
        return ['M', 'race']

    @property
    def continuous(self):
        return [
            'heart rate',
            'mean blood pressure',
            'systemic vascular resistance',
            'hemoglobin',
            'blood urea nitrogen',
            'age',
            'vaso',
            'vent'
        ]

    @property
    def immutables(self):
        return ['M', 'race', 'age']

    @property
    def user_immutables(self):
        return []

    @property
    def monotonic_variables(self):
        return ['vaso', 'vent'], []

    @property
    def target(self):
        return 'diastolic blood pressure Y'

    @property
    def id(self):
        return 'subject_id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        raise NotImplementedError

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        raise NotImplementedError

    def __init__(self, file_path: str):

        data_name: str = 'MIMIC_pressure'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv') # [:10]
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv') # [:10]

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)


class MimicOxygenFeatureSelect(DataCatalog):
    @property
    def categorical(self):
        return []

    @property
    def continuous(self):
        return [
            'heart rate',
            'red blood cell count',
            'sodium',
            'mean blood pressure',
            'systemic vascular resistance',
            'glucose',
            'chloride urine',
            'glascow coma scale total',
            'hematocrit',
            'positive end-expiratory pressure set',
            'respiratory rate',
            'prothrombin time pt',
            'cholesterol',
            'hemoglobin',
            'creatinine',
            'blood urea nitrogen',
            'bicarbonate',
            'calcium ionized',
            'partial pressure of carbon dioxide',
            'magnesium',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
            'calcium urine'
        ]

    @property
    def immutables(self):
        return []

    @property
    def user_immutables(self):
        return []

    @property
    def monotonic_variables(self):
        return [], []

    @property
    def target(self):
        return 'oxygen saturation Y'

    @property
    def id(self):
        return 'subject_id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        raise NotImplementedError

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        raise NotImplementedError

    def __init__(self, file_path: str):

        data_name: str = 'MIMIC_oxygen_feature_select'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)


class MimicOxygen(DataCatalog):
    @property
    def categorical(self):
        return ['M', 'race']

    @property
    def continuous(self):
        return [
            'respiratory rate',
            'mean blood pressure',
            'systemic vascular resistance',
            'hemoglobin',
            'bicarbonate',
            'age',
            'vaso',
            'vent'
        ]

    @property
    def immutables(self):
        return ['M', 'race', 'age']

    @property
    def user_immutables(self):
        return []

    @property
    def monotonic_variables(self):
        return ['vaso', 'vent'], []

    @property
    def target(self):
        return 'oxygen saturation Y'

    @property
    def id(self):
        return 'subject_id'

    @property
    def user_immutables_individual_truth_test(self) -> dict:
        raise NotImplementedError

    def check_user_immutable(self, feature: str, cfs: Union[np.ndarray, torch.Tensor]):
        raise NotImplementedError

    def __init__(self, file_path: str):

        data_name: str = 'MIMIC_oxygen'

        self._file_path: str = file_path + '/'

        df_train: pd.DataFrame = pd.read_csv(self._file_path + 'train_data.csv')
        df_train_new: pd.DataFrame = pd.read_csv(self._file_path + 'train_data_new.csv')

        df_test: pd.DataFrame = pd.read_csv(self._file_path + 'test_data.csv')
        df_test_new: pd.DataFrame = pd.read_csv(self._file_path + 'test_data_new.csv')

        df: pd.DataFrame = pd.concat([df_train, df_train_new], axis=0, ignore_index=True)

        super().__init__(data_name, df, df_train, df_train_new, df_test, df_test_new)
