import numpy as np
import pandas as pd


# series为pandas.series结构,索引从1开始
def nominal_d(series, matrix, matrix_de):
    obj_num = len(series.index)
    for j in range(1, obj_num+1):
        for i in range(j+1, obj_num+1):
            if series[j] == np.NaN or series[i] == np.NaN:
                pass
            else:
                matrix_de[i-1, j-1] += 1
                if series[j] != series[i]:
                    matrix[i-1, j-1] += 1


def asy_binary_d(series, matrix, matrix_de):
    obj_num = len(series.index)
    for j in range(1, obj_num+1):
        for i in range(j+1, obj_num+1):
            if series[j] == np.NaN or series[i] == np.NaN:
                pass
            elif series[j] == 0 and series[i] == 0:
                pass
            else:
                matrix_de[i-1, j-1] += 1
                if series[j] != series[i]:
                    matrix[i-1, j-1] += 1


def numerical_d(series, matrix, matrix_de):
    obj_num = len(series.index)
    max_num = series[1]
    min_num = series[1]
    for k in range(2, obj_num+1):
        if max_num < series[k]:
            max_num = series[k]
        if min_num > series[k]:
            min_num = series[k]
    for j in range(1, obj_num+1):
        for i in range(j+1, obj_num+1):
            if series[j] == np.NaN or series[i] == np.NaN:
                pass
            else:
                matrix_de[i-1, j-1] += 1
                matrix[i-1, j-1] += np.abs(series[i] - series[j]) / (max_num - min_num)


# sequence为pandas.series结构,值从0开始,索引与用户数据中序数属性的原始内容有关
def ordinal_d(series, matrix, matrix_de, sequence):
    obj_num = len(series.index)
    m = len(sequence)
    for i in range(1, obj_num+1):
        if series[i] == np.NaN:
                pass
        else:
            index = series[i]
            series[i] = sequence[index] / (m - 1)
    numerical_d(series, matrix, matrix_de)


def calculate_differential_matrix(columns, series_s, obj_num):
    matrix = np.mat(np.zeros((obj_num, obj_num)))
    matrix_de = np.mat(np.zeros((obj_num, obj_num)))
    for key_, type_ in columns.items():
        if type_ == "1":
            nominal_d(series_s[key_], matrix, matrix_de)
        elif type_ == "2":
            asy_binary_d(series_s[key_], matrix, matrix_de)
        elif type_ == "3":
            numerical_d(series_s[key_], matrix, matrix_de)
        elif type_ == "4":
            print("%s是序数属性,请输入所有状态的排位:\n例如属性状态有大中小三种状态,则输入:小 中 大")
            sequence = input("order: ")
            sequence_new = sequence.split()
            sequence_dict = {}
            i = 0
            for element in sequence_new:
                sequence_dict[element] = i
                i += 1
            ordinal_d(series_s[key_], matrix, matrix_de, sequence_dict)
    for j in range(1, obj_num+1):
        for i in range(j+1, obj_num+1):
            matrix_de[i-1, j-1] = 1 / matrix_de[i-1, j-1]
    return np.multiply(matrix, matrix_de)


print("请输入数据源(csv格式)的路径:")
path = input("path: ")
df = pd.read_csv(path)
print("请指定索引列,输入作为数据索引的列的名称:")
index_name = input("column_name: ")
df = df.set_index([index_name])
obj_nums = len(df.index)
print("请说明每种属性的类型: 1.标称或对称二元属性 2.非对称二元属性 3.数值属性 4.序数属性")
column_list = {}
series_list = {}
for column in df.columns:
    column_type = input("%s: " % column)
    column_list[column] = column_type
    series_ = df[column]
    series_list[column] = series_.copy()
print(calculate_differential_matrix(column_list, series_list, obj_nums))
