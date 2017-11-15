import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calcu_dot_distance(dot1, dot2):
    np_vec1 = dot1.values
    np_vec2 = dot2.values
    return np.linalg.norm(np_vec1 - np_vec2)


# 计算两个类之间的距离,采用最短距离法
def calcu_class_distance(list1, list2):
    min_dot_distance = np.inf
    for dot1 in list1:
        for dot2 in list2:
            d = calcu_dot_distance(dot1, dot2)
            if d < min_dot_distance:
                min_dot_distance = d
    return min_dot_distance


# 寻找到距离矩阵中最小的元素,返回该元素的行列索引,以及该元素值
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
    for k in range(row_number):
        ci = [df_.loc[k]]
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

if __name__ == "__main__":
    # 数据不需要索引列
    print("请输入数据源(csv格式)的路径:")
    path = input("path: ")
    df = pd.read_csv(path)
    print("请输入层次聚类法的阈值:")
    threshold = input("threshold: ")
    mat = clustering(df, float(threshold))
    for b in range(len(mat)):
        print(mat[b])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    plt.rcParams["axes.unicode_minus"] = False
    for c in range(len(mat)):
        for i in mat[c]:
            plt.plot(i["x"], i["y"], mark[c])

    plt.title("hierarchical")
    plt.show()
