from torch.utils.data import Dataset, DataLoader
import torch


class ElasticDataSet(Dataset):
    def __init__(self, *args):
        super(ElasticDataSet, self).__init__()
        assert len(args) >= 1
        self.__data_list: list = [arg for arg in args]
        first_len: int = len(args[0])
        for arg in args:
            assert len(arg) == first_len

    def __len__(self):
        return self.__data_list[0].shape[0]

    def __getitem__(self, idx):
        return [arg[idx] for arg in self.__data_list]



