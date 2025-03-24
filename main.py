# noinspection DuplicatedCode
"""
基于两方多分类的建树。

代码中一些变量和函数名称的命名规则：
1. 位于辅助节点 S_SITE 的变量均以 s_site 开头，后续紧跟下划线，跟作用名称
2. 仅限辅助节点使用的函数，函数名称以 s + 下划线 开头，如 s_check_gini_calculation
3. 算子结算结果均以 v 起始，后跟节点的名称，如 va ，后跟：
    - 两方混合乘法：m ( s2phm )
    - 两方求逆：i ( inv )
    - 两方比较：c ( compare )

启动：

scripts/start grpc-server -c 3

python scripts/python_exec_local.py -f python-extension/secure_tree/two_party_secure_mc-new.py -m org0 org1 org2

scripts/stop
"""

import glob
import json
import os
import pprint
import sys
from typing import Optional, cast

import graphviz
import pandas as pd
from output_style import *

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
# noinspection PyUnresolvedReferences
import motif
import numpy as np

# noinspection PyUnresolvedReferences
from detensor.operators_internal import (
    Compare_2set_internal,
    S2PHM_internal,
    inv_2p_internal,
    dot_internal,
)
import logging

logger = logging.getLogger("Motif")
# setLevel(logging.INFO)

# 以下选项在开发调试时使用，调整显示 numpy 和 pandas 输出的样式
np.set_printoptions(
    threshold=sys.maxsize,
    linewidth=400,
    edgeitems=36,
    # precision=1,# 每个数据保留的小数点位数
)
pd.set_option("display.width", None)  # 自动调整宽度，不会换行
pd.set_option("display.max_columns", None)  # 显示所有列
# pd.set_option("display.max_rows", None)  # 显示所有行


def print_red(param, data):
    print(RED + str(param) + RESET)
    print(data)


def load_excel_data(dataset_path: str) -> pd.DataFrame:
    """加载 excel 格式的数据集"""
    return pd.read_excel(dataset_path, dtype=str)


def detect_invalid_gini_columns(
    dataset: motif.RemoteVariable.RemoteVariable,
) -> dict:
    """
    检查数据集中有可能导致无法计算 gini 值的特征列：
    1. 列中的取值仅有 1 种；
    2. 数据集为空

    参数:
    dataset (motif.RemoteVariable.RemoteVariable): 数据集远程变量

    返回:
    {
        "is_empty": True/False,  # 数据集是否为空
        'cap-color': 'brown',  # 解释: cap-color 列中仅有一种 brown q取值
        ...
    }
    """
    result = {}
    if dataset.empty:
        return {"is_empty": True}
    for column in dataset.columns[:-1]:  # 排除最后一列标签列
        unique_values = dataset[column].unique()
        if len(unique_values) == 1:  # 只有一个唯一值
            result[column] = unique_values[0]  # 将唯一值存入字典
    return result


def s_check_gini_calculation(
    a_dict: motif.RemoteVariable.RemoteVariable,
    b_dict: motif.RemoteVariable.RemoteVariable,
) -> dict:
    """
    对比两个节点发来的可能无法计算 gini 值的字典，返回是否可计算 gini 值的字典。
    如果：两方给出相同列名，且取值相同，则标记为不可计算 gini 值，其余情况，都是可计算的。
    False 为不可计算，Ture 为可计算。

    返回:
    {
        'cap-color': False,
        'gill-color': True,
        ...
    }
    """
    result = {}

    # 检查 is_empty 键
    a_is_empty = a_dict.get("is_empty", False)
    b_is_empty = b_dict.get("is_empty", False)

    if a_is_empty and not b_is_empty:
        # a_dict 为空，则 b_dict 中所有键值对的比较结果都是 False
        return {key: False for key in b_dict.keys()}
    elif b_is_empty and not a_is_empty:
        # b_dict 为空，则 a_dict 中所有键值对的比较结果都是 False
        return {key: False for key in a_dict.keys()}

    # 获取所有的键
    all_keys = set(a_dict.keys()).union(b_dict.keys())
    for key in all_keys:
        if key in a_dict and key in b_dict:
            # 如果键在两个字典中都存在，则比较值
            result[key] = a_dict[key] != b_dict[key]
        else:
            # 只有一个字典包含该键
            result[key] = True

    return result


def scalar_to_vector(value: float) -> np.ndarray:
    """
    将单个数值转换为一维向量
    :param value: Python 原生 int 或 float
    :return: 一维 numpy.ndarray，仅包含该数值
    """
    return np.array([value])  # 创建仅包含该数值的一维数组


def scalar_to_matrix(value) -> np.ndarray:
    """
    将单个数值转换为 NumPy 1x1 矩阵
    :param value: Python 原生 int 或 float
    :return: 1x1 numpy.ndarray
    """
    return np.array([[value]])


def vector_to_diag(vector: list) -> np.ndarray:
    """
    将一维列表转换为对角矩阵
    :param vector: 一维 Python 列表
    :return: 二维 numpy.ndarray，对角线上为 vector 元素，其他元素为 0
    """
    return np.diag(vector)


def matrix_to_vector(matrix: np.ndarray):
    """
    将只有一个元素的矩阵转换为一维向量
    :param matrix: 形状为 (1,1) 的 NumPy 数组
    :return: 一维 numpy.ndarray，仅包含该数值
    """
    return matrix.flatten()  # 将 (1,1) 矩阵展平成一维数组


def matrix_to_diag(matrix: np.ndarray) -> np.ndarray:
    """
    将输入方阵转换为对角矩阵，保留对角线元素，其余元素置为 0
    :param matrix: 二维 NumPy 方阵
    :return: 二维 numpy.ndarray，与输入矩阵大小相同，仅对角线保留原值，其他元素为 0
    """
    diag_values = np.diag(matrix)  # 提取原矩阵的对角线元素
    return np.diag(diag_values)  # 生成新的对角矩阵


def extract_diag_vector(matrix: np.ndarray) -> np.ndarray:
    """
    从输入矩阵中提取对角线元素，生成 NumPy 向量
    :param matrix: 二维 NumPy 数组
    :return: 一维 numpy.ndarray，包含矩阵的对角线元素
    """
    return np.diag(matrix)  # 提取矩阵的对角线元素，返回一维数组


def get_data_by_index(container, index):
    """根据下标返回容器内对应下标的元素"""
    return container[index]


def get_feature_list(dataset: motif.RemoteVariable.RemoteVariable):
    """获取数据集的全部的列名，包括标签列"""
    return dataset.columns


def check_recursion_condition(stats):
    # 停止条件 1：数据集只有标签列（检测是否还有特征列）
    if stats["feature_num"] == 0:
        return False
    return True


# 计算字典所有值的平方和
def sum_of_squares(input_dict: "dict"):
    return sum(value**2 for value in input_dict.values())


# 计算标量的平方
def scalar_square(data: "int"):
    return data**2


def remove_non_computable_gini_columns(
    dataset: motif.RemoteVariable.RemoteVariable,
    column_flags_dict: motif.RemoteVariable.RemoteVariable,
) -> pd.DataFrame:
    """
    参与联合建树的两个节点使用，用于剔除下一次建树时不可用于计算 gini 值的列，
    该列两个节点加起来只有一种取值，对特征划分没有参考价值。

    参数:
    dataset: 传入的当前节点的数据集，用于从其中删除某些不可计算 gini 值的列
    column_flags_dict: 根据辅助节点计算传来的字典，记录了哪些特征是可(True)/不可(False)计算的，根据这个字典处理数据集

    返回:
    新的删除不可计算的特征列的数据集。
    """
    # 遍历字典，如果值为 False，就删除该列
    new_dataset = dataset.copy()  # 创建副本，避免修改原始数据集
    for col, flag in column_flags_dict.items():
        if not flag and col in new_dataset.columns:
            new_dataset = new_dataset.drop(columns=[col])
    return new_dataset


# 根据键获取值
def get_data_by_key(data, key):
    return data[key]


# 矩阵中的每个元素求平方
def matrix_square(matrix):
    return np.square(matrix)


# 矩阵乘以标量
def matrix_multi_scalar(matrix, scalar):
    return np.multiply(matrix, scalar)


# 标量扩展为对角阵
def scalar_to_diag(value: float, size: int) -> np.ndarray:
    """
    将单个数值转换为指定大小的对角矩阵
    :param value: Python 原生 int 或 float，将填充到对角线上的值
    :param size: int，矩阵的大小（行和列数）
    :return: 二维 numpy.ndarray，大小为 size x size 的对角矩阵
    """
    matrix = np.zeros((size, size))  # 创建一个全为 0 的矩阵
    np.fill_diagonal(matrix, value)  # 将对角线上的元素设置为 value
    return matrix


# 列表扩展为对角阵
def list_to_diag(list_data):
    return np.diag(list_data)


# 计算字典中所有矩阵（值）的平方和
def sum_of_squares_dict(data: "dict"):
    """
    传入一个字典，字典中的每个值都是一个 numpy 矩阵，计算每个矩阵的平方，最终求和。
    Returns
    -------
    返回一个矩阵
    """
    return sum(np.square(value) for value in data.values())


# 两个标量相乘
def scalar_multi(scalar1, scalar2):
    return scalar1 * scalar2


# 若干标量相加
def sum_of_scalars(*args):
    return sum(args)


# 若干矩阵相加
def sum_of_matrices(*matrices):
    return np.sum(matrices, axis=0)


# 若干矩阵相乘
def matrix_multi(*args):
    return np.prod(args, axis=0)


def calc_matrix_stats(stats: "dict", feature_name: "str"):
    """
    根据每个节点传入的当前节点的统计量，统计出用于计算当前特征划分后的 gini 值的各种矩阵。

    返回格式：字典

    传入数据：
        stats 为当前节点的全部的初始统计量
        feature_name 为要统计的特征的名称，只会在这个特征中进行统计，其他特征不考虑
    """
    data = stats["feature_dis"][feature_name]
    # 当前特征取值的种类个数，int
    feature_num = len(data.keys())
    # 当前节点全部的样本数，int
    total_samples = stats["total_samples"]
    # 每个特征取值的样本个数，diag
    # 先获取固定的取值情况列表，确保数值顺序对应
    feature_values: list = FEATURES_MAP[feature_name]
    feature_sample_num = np.diag([data[i]["number"].item() for i in feature_values])

    # 每个标签对应当前特征中每个取值的样本个数，dict
    label_matrices = {
        i: np.diag([data[j]["label_dis"][i].item() for j in feature_values])
        for i in ALL_LABELS
    }
    # 当前节点样本个数，diag
    total_samples_matrix = np.diag([total_samples.item()] * feature_num)

    # 计算 非 的情况
    diag_elements = np.diag(feature_sample_num)  # 获取原对角线元素
    new_diag = np.sum(diag_elements) - diag_elements  # 计算新的对角线元素
    feature_sample_num_w = np.diag(new_diag)  # 生成新的对角阵

    label_matrices_w = {
        i: np.diag(np.sum(np.diag(M)) - np.diag(M)) for i, M in label_matrices.items()
    }

    # 组合成字典返回
    res = {
        "feature_sample_num": feature_sample_num,
        "feature_sample_num-w": feature_sample_num_w,  # w: without
        "label_matrices": label_matrices,
        "label_matrices-w": label_matrices_w,
        "total_samples_matrix": total_samples_matrix,
    }
    # print(GREEN + "-" * 50 + "calc_matrix_stats:" + "-" * 50 + RESET)
    # pprint.pprint(res)
    return res


def get_next_feature_index(
    stats_a: motif.RemoteVariable.RemoteVariable,
    stats_b: motif.RemoteVariable.RemoteVariable,
):
    # 计算划分前的 gini 值
    # 分子部分
    # 本地计算
    a_lo_label_sum = motif.RemoteCallAt(A_SITE)(sum_of_squares)(
        stats_a["label_dis"]
    )  # A 不同标签样本个数的平方和
    b_lo_label_sum = motif.RemoteCallAt(B_SITE)(sum_of_squares)(
        stats_b["label_dis"]
    )  # B 不同标签样本个数的平方和

    # vdat + vdbt = sum(n_Ai * n_Bi)
    vda = 0
    vdb = 0
    vt_list = []
    for label in ALL_LABELS:
        at = stats_a["label_dis"][label]
        bt = stats_b["label_dis"][label]
        vdat, vdbt = dot_internal(at, bt, S_SITE)
        vt_list.append([vdat, vdbt])

    for i in range(len(ALL_LABELS)):
        vda += vt_list[i][0]
        vdb += vt_list[i][1]
    # va1 + vb1 = 分子
    va1 = a_lo_label_sum + 2 * vda
    vb1 = b_lo_label_sum + 2 * vdb
    # print("分子:\t", va1.at_public().evaluate() + vb1.at_public().evaluate())

    # 分母部分
    # 本地计算
    a_lo_sample_square = motif.RemoteCallAt(A_SITE)(scalar_square)(
        stats_a["total_samples"]
    )  # sum(nA^2)
    b_lo_sample_square = motif.RemoteCallAt(B_SITE)(scalar_square)(
        stats_b["total_samples"]
    )  # sum(nB^2)

    # vda2 + vbd2 = nA * nB
    vda2, vdb2 = dot_internal(
        stats_a["total_samples"], stats_b["total_samples"], S_SITE
    )

    # va2 + vb2 = 分母
    va2, vb2 = inv_2p_internal(
        a_lo_sample_square + 2 * vda2, b_lo_sample_square + 2 * vdb2, S_SITE
    )
    # print("分母:\t", va2.at_public().evaluate() + vb2.at_public().evaluate())

    # 划分前的 gini 值 = 1 - (va_gini_before + vb_gini_before)
    va_gini_before, vb_gini_before = S2PHM_internal([va1, va2], [vb1, vb2], S_SITE)
    print(
        "\033[35m\033[106m划分前的 gini 值:\033[0m\t",
        1
        - (
            va_gini_before.at_public().evaluate()
            + vb_gini_before.at_public().evaluate()
        ),
    )

    # 计算划分后的 gini 值
    # 当前所有特征，A B 相等，从 A 公开
    feature_list = stats_a["feature_list"].at_public().evaluate()
    # feature_list: ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    for feature in feature_list:
        print(CYAN + f"当前特征：{feature}" + RESET)

        a_matrices = motif.RemoteCallAt(A_SITE)(calc_matrix_stats)(stats_a, feature)
        b_matrices = motif.RemoteCallAt(B_SITE)(calc_matrix_stats)(stats_b, feature)
        # ######################### 第一部分计算 #########################
        # 1.1 当前特征中，每个取值的样本个数矩阵
        a_matrix_1 = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
            a_matrices, "feature_sample_num"
        )  # $n_{A_p}$  ✅
        b_matrix_1 = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
            b_matrices, "feature_sample_num"
        )  # ✅
        # 1.2 上述两矩阵进行两方隐私乘法
        va_1, vb_1 = dot_internal(a_matrix_1, b_matrix_1, S_SITE)  # ✅
        # 结果分别本地乘以 2 ，满足最终结果乘以 2 的效果
        va_1 = motif.RemoteCallAt(A_SITE)(matrix_multi_scalar)(va_1, 2)
        vb_1 = motif.RemoteCallAt(B_SITE)(matrix_multi_scalar)(vb_1, 2)

        # 1.2 中矩阵本地求平方
        a_matrix_square_1 = motif.RemoteCallAt(A_SITE)(matrix_square)(
            a_matrix_1
        )  # $n_{A_p}^2$  ✅
        b_matrix_square_1 = motif.RemoteCallAt(B_SITE)(matrix_square)(b_matrix_1)  # ✅
        # 1.3 标签对应的特征的矩阵
        a_label_matrices = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
            a_matrices, "label_matrices"
        )

        a_matrix_2 = motif.RemoteCallAt(A_SITE)(sum_of_squares_dict)(
            a_label_matrices
        )  # $\sum_{i=1}^m n_{A_{p_i}}^2$  ✅
        b_label_matrices = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
            b_matrices, "label_matrices"
        )
        b_matrix_2 = motif.RemoteCallAt(B_SITE)(sum_of_squares_dict)(
            b_label_matrices
        )  # ✅

        # 1.4 当前节点拥有的样本总数
        a_matrix_3 = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
            a_matrices, "total_samples_matrix"
        )  # $n_A$  ✅
        b_matrix_3 = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
            b_matrices, "total_samples_matrix"
        )

        # 1.5 本地矩阵相乘
        a_matrix_4 = motif.RemoteCallAt(A_SITE)(matrix_multi)(
            a_matrix_3, a_matrix_1
        )  # ✅
        b_matrix_4 = motif.RemoteCallAt(B_SITE)(matrix_multi)(
            b_matrix_3, b_matrix_1
        )  # ✅

        # 1.6 两方矩阵乘法
        va_2, vb_2 = dot_internal(a_matrix_3, b_matrix_1, S_SITE)  # ✅
        va_3, vb_3 = dot_internal(a_matrix_1, b_matrix_3, S_SITE)  # ✅

        # 1.7 两方隐私乘法计算标签对应的特征的矩阵的乘积
        va = 0
        vb = 0
        for label in ALL_LABELS:
            a_tmp_label_matrix = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
                a_label_matrices, label
            )
            b_tmp_label_matrix = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
                b_label_matrices, label
            )
            # 两方隐私乘法
            va_tmp, vb_tmp = dot_internal(
                a_tmp_label_matrix, b_tmp_label_matrix, S_SITE
            )
            va += va_tmp
            vb += vb_tmp
        # 分别乘以 2  ✅
        va = motif.RemoteCallAt(A_SITE)(matrix_multi_scalar)(va, 2)
        vb = motif.RemoteCallAt(B_SITE)(matrix_multi_scalar)(vb, 2)

        # 🔼 计算分子部分
        a_numer = motif.RemoteCallAt(A_SITE)(sum_of_matrices)(
            a_matrix_square_1, va_1, -a_matrix_2, -va
        )
        b_numer = motif.RemoteCallAt(B_SITE)(sum_of_matrices)(
            b_matrix_square_1, vb_1, -b_matrix_2, -vb
        )

        # 🔽 计算分母部分  ✅
        a_denom = motif.RemoteCallAt(A_SITE)(sum_of_matrices)(a_matrix_4, va_2, va_3)
        b_denom = motif.RemoteCallAt(B_SITE)(sum_of_matrices)(b_matrix_4, vb_2, vb_3)
        print("计算分母部分")
        print(a_denom.at_public().evaluate() + b_denom.at_public().evaluate())
        # 计算分母的逆矩阵，两方求逆
        a_inv_denom, b_inv_denom = inv_2p_internal(a_denom, b_denom, S_SITE)

        # ✴️ 两方混合乘法
        a_gini_after1, b_gini_after1 = S2PHM_internal(
            [a_numer, a_inv_denom], [b_numer, b_inv_denom], S_SITE
        )

        # ######################### 第二部分计算 #########################
        # 2.1 当前特征中，每个取值的样本个数矩阵
        a_matrix_w_1 = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
            a_matrices, "feature_sample_num-w"
        )  # $n_{A_\hat p}$
        b_matrix_w_1 = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
            b_matrices, "feature_sample_num-w"
        )
        # 2.2 上述两矩阵进行两方隐私乘法
        va_w_1, vb_w_1 = dot_internal(a_matrix_w_1, b_matrix_w_1, S_SITE)
        # 结果分别本地乘以 2 ，满足最终结果乘以 2 的效果
        va_w_1 = motif.RemoteCallAt(A_SITE)(matrix_multi_scalar)(va_w_1, 2)
        vb_w_1 = motif.RemoteCallAt(B_SITE)(matrix_multi_scalar)(vb_w_1, 2)

        # 2.2 中矩阵本地求平方
        a_matrix_square_w_1 = motif.RemoteCallAt(A_SITE)(matrix_square)(
            a_matrix_w_1
        )  # $n_{A_\hat p}^2$
        b_matrix_square_w_1 = motif.RemoteCallAt(B_SITE)(matrix_square)(b_matrix_w_1)

        # 2.3 标签对应的特征的矩阵
        a_label_matrices_w = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
            a_matrices, "label_matrices-w"
        )
        a_matrix_w_2 = motif.RemoteCallAt(A_SITE)(sum_of_squares_dict)(
            a_label_matrices_w
        )  # $\sum_{i=1}^m n_{A_{\hat p_i}}^2$
        b_label_matrices_w = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
            b_matrices, "label_matrices-w"
        )
        b_matrix_w_2 = motif.RemoteCallAt(B_SITE)(sum_of_squares_dict)(
            b_label_matrices_w
        )

        # 2.4 本地矩阵相乘
        a_matrix_4 = motif.RemoteCallAt(A_SITE)(matrix_multi)(a_matrix_3, a_matrix_w_1)
        b_matrix_4 = motif.RemoteCallAt(B_SITE)(matrix_multi)(b_matrix_3, b_matrix_w_1)

        # 2.5 两方矩阵乘法
        va_2, vb_2 = dot_internal(a_matrix_3, b_matrix_w_1, S_SITE)
        va_3, vb_3 = dot_internal(a_matrix_w_1, b_matrix_3, S_SITE)

        # 2.6 两方隐私乘法计算标签对应的特征的矩阵的乘积
        va = 0
        vb = 0
        for label in ALL_LABELS:
            a_tmp_label_matrix = motif.RemoteCallAt(A_SITE)(get_data_by_key)(
                a_label_matrices_w, label
            )
            b_tmp_label_matrix = motif.RemoteCallAt(B_SITE)(get_data_by_key)(
                b_label_matrices_w, label
            )
            # 两方隐私乘法
            va_tmp, vb_tmp = dot_internal(
                a_tmp_label_matrix, b_tmp_label_matrix, S_SITE
            )
            va += va_tmp
            vb += vb_tmp
        # 分别乘以 2
        va = motif.RemoteCallAt(A_SITE)(matrix_multi_scalar)(va, 2)
        vb = motif.RemoteCallAt(B_SITE)(matrix_multi_scalar)(vb, 2)

        # 🔼 计算分子部分
        a_numer = motif.RemoteCallAt(A_SITE)(sum_of_matrices)(
            a_matrix_square_w_1, va_w_1, -a_matrix_w_2, -va
        )
        b_numer = motif.RemoteCallAt(B_SITE)(sum_of_matrices)(
            b_matrix_square_w_1, vb_w_1, -b_matrix_w_2, -vb
        )
        # 🔽 计算分母部分
        a_denom = motif.RemoteCallAt(A_SITE)(sum_of_matrices)(a_matrix_4, va_2, va_3)
        b_denom = motif.RemoteCallAt(B_SITE)(sum_of_matrices)(b_matrix_4, vb_2, vb_3)

        # 计算分母的逆矩阵，两方求逆
        a_inv_denom, b_inv_denom = inv_2p_internal(a_denom, b_denom, S_SITE)

        # ✴️ 两方混合乘法
        a_gini_after2, b_gini_after2 = S2PHM_internal(
            [a_numer, a_inv_denom], [b_numer, b_inv_denom], S_SITE
        )

        # 最终：A B 分别持有直接计算划分后的 gini 值的数据
        # A: a_gini_after1, a_gini_after2 ，B: b_gini_after1, b_gini_after2
        a_gini_after = motif.RemoteCallAt(A_SITE)(sum_of_matrices)(
            a_gini_after1, a_gini_after2
        )
        b_gini_after = motif.RemoteCallAt(B_SITE)(sum_of_matrices)(
            b_gini_after1, b_gini_after2
        )

        print(BLUE + "划分后的 gini 值，特征 {} ".format(feature) + RESET)
        print(a_gini_after.at_public().evaluate() + b_gini_after.at_public().evaluate())
    exit()


def calculate_statistics(dataset: pd.DataFrame) -> dict:
    """
    计算数据集中的统计量，返回格式为字典。

    参数：
    dataset: pandas.DataFrame
        一个 DataFrame，最后一列是标签列，每个特征列有 2 种以上的取值。

    ALL_LABELS: list
        所有可能的标签值

    返回：
    type: dict
        content：
        - 'total_samples' : int，样本数量
        - 'label_dis' : dict，标签的不同取值对应样本的个数
        - 'feature_dis' : dict，每个特征值的样本数量和标签分布
        - 'feature_number' : int，特征数量
    """
    total_samples = np.array([[len(dataset)]])  # 样本总数
    # 获取标签列的分布情况，并确保所有标签都显示，即使为 0
    label_dis = dataset.iloc[:, -1].value_counts().to_dict()
    # 确保所有标签都有显示，未出现的标签设置为 0
    label_dis = {
        label: np.array([[label_dis.get(label, 0)]], dtype=int) for label in ALL_LABELS
    }
    # 初始化 feature_dis
    feature_dis = {}
    # 统计每个特征的取值分布情况
    for feature, possible_values in FEATURES_MAP.items():  # 遍历每个特征及其可能的取值
        feature_dis[feature] = {}

        for value in possible_values:  # 对每个特征的所有可能取值进行统计
            # 获取该特征取值对应的数据子集
            subset = dataset[dataset[feature] == value]

            # 获取该子集在各个标签下的分布情况
            value_label_dis = subset.iloc[:, -1].value_counts().to_dict()

            # 确保所有标签都有显示，未出现的标签设置为 0
            value_label_dis = {
                label: np.array([[value_label_dis.get(label, 0)]], dtype=int)
                for label in ALL_LABELS
            }

            # 保存特征取值的统计信息
            feature_dis[feature][value] = {
                "number": np.array([[len(subset)]], dtype=int),  # 样本个数
                "label_dis": value_label_dis,  # 标签分布
            }

        # 确保所有特征取值（即使为 0 样本）都显示
        for value in possible_values:
            if value not in feature_dis[feature]:
                feature_dis[feature][value] = {
                    "number": np.array([[0]], dtype=int),  # 样本个数为 0
                    "label_dis": {
                        label: np.array([[0]], dtype=int)  # 标签分布为 0
                        for label in ALL_LABELS
                    },
                }
    # 输出统计信息
    stats = {
        "total_samples": total_samples,
        "label_dis": label_dis,
        "feature_dis": feature_dis,
        "feature_list": list(feature_dis.keys()),
        "feature_number": np.array([[len(dataset.columns) - 1]]),  # 特征数量
    }

    return stats


def build_tree(
    a_dataset: motif.RemoteVariable.RemoteVariable,
    b_dataset: motif.RemoteVariable.RemoteVariable,
    current_node: Optional["Node"],
    parent_node: Optional["Node"],
):
    global root

    # 计算 A B 数据集统计量
    a_stats = motif.RemoteCallAt(A_SITE)(calculate_statistics)(a_dataset)
    b_stats = motif.RemoteCallAt(B_SITE)(calculate_statistics)(b_dataset)
    # print(RED + "A 统计量：" + RESET)
    # pprint.pprint(a_stats.at_public().evaluate())
    # exit()
    # print(BLUE + "-" * 150 + RESET)
    # print(RED + exit()"B 统计量：" + RESET)
    # pprint.pprint(b_stats.at_public().evaluate())
    # # 根据两个统计量计算划分特征 gini 值
    next_feature_index = get_next_feature_index(a_stats, b_stats)


if __name__ == "__main__":
    A_SITE = motif.Site(1)
    B_SITE = motif.Site(2)
    S_SITE = motif.Site(3)  # 辅助节点

    class Node:
        def __init__(self):
            self.feature_name = None
            self.lc = None
            self.rc = None
            self.parent = None

            self.is_leaf = False
            self.pred = None  # is_leaf 生效时有效，标记预测结果

    # 创建输出文件夹
    os.makedirs("/usr/src/demo/python-extension/output", exist_ok=True)

    # dataset_path 给出的规则为数据集所在目录，不需要文件名
    # 确保给出目录下存在 A B 节点持有的文件，格式为 xxx-A.xlsx
    dataset_path = "python-extension/dataset/CLS/MC/car"

    # 打开描述文件
    with open(dataset_path + "/DESC.json", "r", encoding="utf-8") as f:
        desc_json = json.load(f)
    ALL_LABELS = desc_json["ALL_LABELS"]
    FEATURES_MAP = desc_json["FEATURES_MAP"]

    # 生成两个文件的路径
    file_A = glob.glob(os.path.join(dataset_path, "*-A.xlsx"))[0]
    file_B = glob.glob(os.path.join(dataset_path, "*-B.xlsx"))[0]

    # 读取两个 Excel 文件
    dataset_a = motif.RemoteCallAt(A_SITE)(load_excel_data)(file_A)
    dataset_b = motif.RemoteCallAt(B_SITE)(load_excel_data)(file_B)

    # 生成空根节点
    root = None

    build_tree(
        a_dataset=dataset_a,
        b_dataset=dataset_b,
        current_node=None,
        parent_node=None,
    )

"""
scripts/start grpc-server -c 3

python scripts/python_exec_local.py -f python-extension/secure_tree/two_party_secure_mc-new.py -m org0 org1 org2

scripts/stop
"""
