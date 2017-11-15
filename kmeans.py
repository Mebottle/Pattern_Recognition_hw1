import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calcu_distance(dot, center):
    np_vec1 = dot
    np_vec2 = center
    return np.linalg.norm(np_vec1 - np_vec2)


def kmeans(df_, k):
    obj_num = len(df_.index)
    obj_belong = np.array(np.zeros(obj_num))
    centers = []
    for a in range(k):
        cj = np.array(df_.loc[a].values)
        centers.append(cj)
    is_change = True

    while is_change:
        is_change = False
        for i in range(obj_num):
            min_dot_center = np.inf
            min_index = 0
            for j in range(k):
                if not len(centers[j]):
                    continue
                d = calcu_distance(df_.loc[i].values, centers[j])
                if d < min_dot_center:
                    min_dot_center = d
                    min_index = j
            if obj_belong[i] != min_index:
                obj_belong[i] = min_index
                is_change = True
        for h in range(k):
            index_set_h = np.where(obj_belong == h)[0]
            list_ = []
            for ind in index_set_h:
                list_.append(df_.loc[ind].values)
            new_center_vec = np.mean(np.mat(list_), axis=0)
            centers[h] = np.array(new_center_vec)[0]
    return obj_belong

if __name__ == "__main__":
    # 数据不需要索引列
    print("请输入数据源(csv格式)的路径:")
    path = input("path: ")
    df = pd.read_csv(path)

    print("请输入k值:")
    k_ = int(input("k: "))

    belongs = kmeans(df, k_)
    for c in range(k_):
        indexes = np.where(belongs == c)[0]
        print("class %d : " % c)
        for ind in indexes:
            print(df.loc[[ind]])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    plt.rcParams["axes.unicode_minus"] = False
    for c in range(3):
        indexes = np.where(belongs == c)[0]
        for i in indexes:
            plt.plot(df.loc[i]["x"], df.loc[i]["y"], mark[c])
    plt.title("k-means")
    plt.show()
