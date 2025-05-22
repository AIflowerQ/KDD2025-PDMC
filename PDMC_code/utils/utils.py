import random

import numpy as np
import torch.nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel


def plot_column_distributions(tensor, bins=20, figsize=(8, 6), color='skyblue', edgecolor='black',
                                       title_prefix='Column'):
    """
    为二维数组的每一列生成单独的分布图，并依次显示。

    参数:
    - tensor (np.ndarray): 输入的二维数组。
    - bins (int): 直方图的箱子数目，默认20。
    - figsize (tuple): 图形大小，默认为 (8, 6)。
    - color (str): 直方图的填充颜色，默认为 'skyblue'。
    - edgecolor (str): 直方图条形的边缘颜色，默认为 'black'。
    - title_prefix (str): 子图标题的前缀，默认 'Column'。

    返回:
    - None
    """
    num_cols = tensor.shape[1]  # 获取列数

    # 遍历每一列并绘制单独的直方图
    for i in range(num_cols):
        plt.figure(figsize=figsize)  # 每列一个新图
        plt.hist(tensor[:, i].cpu().numpy(), bins=bins, color=color, edgecolor=edgecolor)  # 绘制直方图
        plt.title(f'{title_prefix} {i + 1}')  # 设置标题
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()  # 自动调整布局
        plt.show()  # 显示当前图


def get_torch_model_device(model: torch.nn.Module):
    return next(model.parameters()).device


def get_torch_tensor_device(tensor: torch.Tensor):
    return tensor.device


def merge_dict(dict_list: list, merge_func, **func_args):
    # assert len(dict_list) > 1, 'len(dict_list) should bigger than 1'
    first_dict: dict = dict_list[0]
    keys = first_dict.keys()
    for element_dict in dict_list:
        assert keys == element_dict.keys()

    result: dict = dict()
    for key in keys:
        elements_list: list = [element_dict[key] for element_dict in dict_list]
        result[key] = merge_func(elements_list, **func_args)

    return result


def mean_merge_dict_func(elements_list, **args):
    # print(args)
    return float(np.mean(elements_list))


def sum_and_mean_merge_dict_func(elements_list, **args):
    # print(args)
    return float(np.sum(elements_list) / args['total_num'])


def show_me_a_list_func(elements_list, **args):
    # print(args)
    return elements_list

def two_level_mean_merge_dict_func(elements_list, **args):
    if isinstance(elements_list[0], dict):
        return merge_dict(elements_list, mean_merge_dict_func)
    else:
        return mean_merge_dict_func(elements_list)


def show_me_all_the_fucking_result(raw_result: dict, metric_list: list, k_list: list, best_index: int) -> dict:
    result_dict: dict = dict()
    for metric in metric_list:
        for k in k_list:
            temp_array: np.array = np.array(merge_dict(raw_result[metric], show_me_a_list_func)[k])
            dict_key: str = str(metric) + '@' + str(k)
            result_dict[dict_key] = temp_array[best_index]
    return result_dict


def show_me_all_the_fucking_explicit_result(raw_result: dict, metric_list: list, best_index: int) -> dict:
    result_dict: dict = dict()
    for metric in metric_list:
        result_dict[metric] = raw_result[metric][best_index]
    return result_dict

def transfer_loss_dict_to_line_str(loss_dict: dict) -> str:
    result_str: str = ''
    for key in loss_dict.keys():
        result_str += (str(key) + ': ' + str(loss_dict[key]) + ', ')

    result_str = result_str[0:len(result_str)-2]
    return result_str

def plot_distributions(df):
    for col in df.columns:
        plt.figure(figsize=(8, 6))  # 设置单独图的大小
        sns.histplot(df[col], bins=10, kde=True, color='blue')  # 直方图+密度图
        plt.title(f'Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


def set_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def compute_mmd(X, Y, gamma=1.0):
    """
    计算两组数据 X 和 Y 的 MMD（最大均值差异）

    X, Y: 样本数据，形状为 (n_samples, n_features)
    gamma: 高斯核的宽度参数

    返回 MMD 值
    """
    XX = rbf_kernel(X, X, gamma=gamma)  # X 之间的核矩阵
    YY = rbf_kernel(Y, Y, gamma=gamma)  # Y 之间的核矩阵
    XY = rbf_kernel(X, Y, gamma=gamma)  # X 和 Y 之间的核矩阵

    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return mmd

def ks_2samp_each_column(a1, a2):

    columns_num: int = a1.shape[1]

    for i in range(columns_num):
        print("column", i + 1)
        # print(a1[:, i].reshape(-1))
        # print(a2[:, i].reshape(-1))
        print(ks_2samp(a1[:, i].reshape(-1), a2[:, i].reshape(-1)))


def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))

def mahalanobis_distance_torch(x, mean, cov_inv):
    diff = x - mean
    return torch.sqrt(torch.sum(torch.matmul(diff, cov_inv) * diff, dim=1))


