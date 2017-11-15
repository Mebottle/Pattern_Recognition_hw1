import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calcu_distance(dot1, dot2):
    np_vec1 = dot1.values
    np_vec2 = dot2.values
    return np.linalg.norm(np_vec1 - np_vec2)


def max_min_dist_clustering(df_):
    dots = []
    row_number = len(df_.index)
    for i in range(row_number):
        dots.append(df_.loc[i])

    # step 1
    centers = [dots[0].copy()]

    # step 2
    max_distance = 0
    max_dot = 0
    for dot in dots:
        d = calcu_distance(dot, centers[0])
        if d > max_distance:
            max_distance = d
            max_dot = dot.copy()
    centers.append(max_dot)
    threshold_t = (1 / 2) * calcu_distance(centers[0], centers[1])

    # step 3, 4 and 5
    while True:
        max_min_distance = 0
        center_candidate = 0
        for dot in dots:
            min_distance = np.inf
            for center in centers:
                d = calcu_distance(dot, center)
                if d < min_distance:
                    min_distance = d
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                center_candidate = dot.copy()
        if max_min_distance > threshold_t:
            centers.append(center_candidate)
        else:
            break

    # step 6
    dict_ = []
    for l in range(len(centers)):
        ci = []
        dict_.append(ci)
    for dot in dots:
        min_dot_center_d = np.inf
        dot_class_index = 0
        for i in range(len(centers)):
            d = calcu_distance(dot, centers[i])
            if d < min_dot_center_d:
                min_dot_center_d = d
                dot_class_index = i
        dict_[dot_class_index].append(dot.copy())
    return dict_

if __name__ == "__main__":
    # 数据不需要索引列
    print("请输入数据源(csv格式)的路径:")
    path = input("path: ")
    df = pd.read_csv(path)
    dic = max_min_dist_clustering(df)

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    plt.rcParams["axes.unicode_minus"] = False
    for c in range(len(dic)):
        print("class %d" % c)
        for i in dic[c]:
            print(i)
            plt.plot(i["x"], i["y"], mark[c])
        print("\n")
    plt.title("max_min distance")
    plt.show()
