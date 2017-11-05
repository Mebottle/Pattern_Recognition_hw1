import numpy as np
import pandas as pd


def calcu_dot_distance(dot1, dot2):
    np_vec1 = np.array(dot1.values)
    np_vec2 = np.array(dot2.values)
    return np.linalg.norm(np_vec1 - np_vec2)


def calcu_class_distance(list1, list2):
    min_dot_distance = np.inf
    for dot1 in list1:
        for dot2 in list2:
            d = calcu_dot_distance(dot1, dot2)
            if d < min_dot_distance:
                min_dot_distance = d
    return min_dot_distance


def find_min_class_distance(matrix_):
    min_class_distance = np.inf
    x = 0
    y = 0
    for i in range(len(matrix_)):
        for j in range(len(matrix_[i])):
            if i != j and matrix_[i][j] < min_class_distance:
                min_class_distance = matrix_[i][j]
                x = i
                y = j
    return x, y, min_class_distance


def clustering(df_, t):
    classes_ = []
    row_number = len(df_.index)
    for k in range(1, row_number+1):
        ci = [df_.loc[[k]]]
        classes_.append(ci)

    matrix_ = []
    for i in classes_:
        mi = []
        for j in classes_:
            mi.append(calcu_class_distance(i, j))
        matrix_.append(mi)

    m = matrix_
    while True:
        x, y, min_d = find_min_class_distance(m)
        if min_d > t:
            break
        classes_[x].extend(classes_[y])
        classes_.pop(y)
        m = []
        for i in classes_:
            mi = []
            for j in classes_:
                mi.append(calcu_class_distance(i, j))
            m.append(mi)
    return classes_

print("请输入数据源(csv格式)的路径:")
path = input("path: ")
df = pd.read_csv(path)
print("请指定索引列,输入作为数据索引的列的名称:")
index_name = input("column_name: ")
df = df.set_index([index_name])
print("请输入层次聚类法的阈值:")
threshold = input("threshold: ")
mat = clustering(df, float(threshold))
for b in range(len(mat)):
    print(mat[b])
