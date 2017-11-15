import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Args:

    def __init__(self, exp_k, least_n, max_its, max_pair, theta_s, theta_c):
        self.exp_k = exp_k          # 希望的聚类中心数
        self.least_n = least_n      # 聚类中的最少样本数
        self.max_its = max_its      # 最大迭代次数
        self.max_pair = max_pair    # 每一次迭代中允许合并的聚类中心的最大对数
        self.theta_s = theta_s      # 标准差阈值,用于分裂
        self.theta_c = theta_c      # 两聚类中心的最短距离,用于合并


class Cluster:

    def __init__(self, sample_num, center, inner_avg_distance, sigma, df):
        self.sample_num = sample_num                    # 类中样本数
        self.center = center                            # 聚类中心向量
        self.inner_avg_distance = inner_avg_distance    # 类内平均距离
        self.sigma = sigma                              # 聚类的样本的标准差向量
        self.dots = []                                  # 属于该聚类的样本点的索引集合
        self.df = df                                    # 存储样本数据的dataframe

    # 计算聚类中心向量
    def calcu_center(self):
        list_ = []
        for ind in self.dots:
            list_.append(self.df.loc[ind].values)
        self.center = np.array(np.mean(np.mat(list_), axis=0))[0]

    # 计算类内平均距离
    def calcu_inner_d(self):
        list_ = []
        for ind in self.dots:
            vec_dot = self.df.loc[ind].values
            list_.append(np.linalg.norm(vec_dot - self.center))
        self.inner_avg_distance = np.mean(list_)

    # 计算标准差向量
    def calcu_std(self):
        list_ = []
        for ind in self.dots:
            list_.append(self.df.loc[ind].values)
        self.sigma = np.array(np.std(np.mat(list_), axis=0))[0]


class CenterDistance:
    def __init__(self, d, i, j):
        self.d = d
        self.i = i
        self.j = j


# 计算一个样本和一个聚类中心的欧氏距离
def calcu_distance(dot, center):
    np_vec1 = dot.values
    np_vec2 = center
    return np.linalg.norm(np_vec1 - np_vec2)


# 寻找距离样本最近的聚类中心,并返回该聚类中心的索引
def find_index(dot, list_center):
    min_dot_center = np.inf
    min_index = 0
    for center_j in range(len(list_center)):
        d = calcu_distance(dot, list_center[center_j].center)
        if d < min_dot_center:
                min_dot_center = d
                min_index = center_j
    return min_index


def isodata(df, n_c_, exp_k, least_n, max_its, max_pair, theta_s, theta_c):
    list_c = []              # 存储所有的cluster
    sigma_k = 0.5            # 分裂常数
    n_c = n_c_               # 聚类中心个数
    obj_num = len(df.index)  # 样本数

    # step 1
    args = Args(exp_k, least_n, max_its, max_pair, theta_s, theta_c)
    for i in range(n_c):
        cluster = Cluster(0, df.loc[i].values, 0, 0, df)
        list_c.append(cluster)

    iter_num = 0
    while True:
        # step 2
        for c in list_c:
            c.sample_num = 0
            c.inner_avg_distance = 0
            c.sigma = 0
            c.dots = []
        for sample_i in range(obj_num):
            min_index = find_index(df.loc[[sample_i]], list_c)
            list_c[min_index].dots.append(sample_i)
            list_c[min_index].sample_num += 1

        # step 3
        center_j_copy = 0
        for center_j in range(n_c):
            if list_c[center_j_copy].sample_num < args.least_n:
                dots = list_c[center_j_copy].dots.copy()
                del list_c[center_j_copy]
                center_j_copy -= 1
                n_c -= 1
                for sample_i in dots:
                    min_index = find_index(df.loc[[sample_i]], list_c)
                    list_c[min_index].dots.append(sample_i)
                    list_c[min_index].sample_num += 1
            center_j_copy += 1
        # step 4 5
        for center_j in range(n_c):
            list_c[center_j].calcu_center()
            list_c[center_j].calcu_inner_d()

        # step 6
        total_avg_distance = 0
        for center_j in range(n_c):
            total_avg_distance += list_c[center_j].sample_num * list_c[center_j].inner_avg_distance
        total_avg_distance /= obj_num

        # step 7
        if iter_num >= args.max_its:
            break  # 算法结束
        is_split = False

        if n_c <= args.exp_k/2 or (n_c < 2*args.exp_k and iter_num % 2 == 1):
            # step 8
            for center_j in range(n_c):
                list_c[center_j].calcu_std()

            # step 9
            max_sigmas = []     # 存所有的sigma_max
            max_indexes = []    # 存每个sigma_max在sigma向量中的位置
            for center_j in range(n_c):
                max_sigma = 0
                max_index = 0
                for sigma_index in range(len(list_c[center_j].sigma)):
                    if list_c[center_j].sigma[sigma_index] > max_sigma:
                        max_sigma = list_c[center_j].sigma[sigma_index]
                        max_index = sigma_index
                max_sigmas.append(max_sigma)
                max_indexes.append(max_index)

            # step 10 分裂操作
            center_j_copy = 0
            for center_j in range(n_c):
                if max_sigmas[center_j_copy] > args.theta_s:
                    if (list_c[center_j_copy].inner_avg_distance > total_avg_distance and list_c[center_j_copy].sample_num > 2*(args.least_n+1)) or n_c <= args.exp_k/2:
                        new_center_1 = list_c[center_j_copy].center.copy()
                        new_center_1[max_indexes[center_j_copy]] += sigma_k * max_sigmas[center_j_copy]
                        new_center_2 = list_c[center_j_copy].center.copy()
                        new_center_2[max_indexes[center_j_copy]] -= sigma_k * max_sigmas[center_j_copy]
                        cluster_new_1 = Cluster(0, new_center_1, 0, 0, df)
                        cluster_new_2 = Cluster(0, new_center_2, 0, 0, df)
                        del list_c[center_j_copy]
                        center_j_copy -= 1
                        list_c.append(cluster_new_1)
                        list_c.append(cluster_new_2)
                        n_c += 1
                        is_split = True
                        break  # 分裂一次后停止
                center_j_copy += 1

        if is_split:
            continue

        if n_c >= 2*args.exp_k or (n_c > args.exp_k/2 and iter_num%2 == 0) or (not is_split):
            center_distances = []
            # step 11
            for n in range(n_c-1):
                for m in range(n+1, n_c):
                    d = np.linalg.norm(list_c[n].center - list_c[m].center)
                    if d < args.theta_c:
                        c_d = CenterDistance(d, n, m)
                        center_distances.append(c_d)

            # step 12
            center_distances.sort(key=lambda x: x.d, reverse=False)
            merge_record = []
            merge_times = 0
            delate_list = []
            for c_d in center_distances:
                if merge_times > args.max_pair:
                    break
                if c_d.i not in merge_record and c_d.j not in merge_record:
                    # step 13
                    center_1 = list_c[c_d.i].center.copy()
                    center_2 = list_c[c_d.j].center.copy()
                    n_i = list_c[c_d.i].sample_num
                    n_j = list_c[c_d.j].sample_num
                    new_center = (n_i * center_1 + n_j * center_2) * (1 / (n_i + n_j))
                    delate_list.append(c_d.i)
                    delate_list.append(c_d.j)
                    c = Cluster(0, new_center, 0, 0, df)
                    list_c.append(c)
                    n_c -= 1
                    merge_times += 1
                    merge_record.append(c_d.i)
                    merge_record.append(c_d.j)
            list_c_2 = []
            for ind in delate_list:
                list_c_2.append(list_c[ind])
            list_c = list(set(list_c).difference(set(list_c_2)))

        # step 14
        if iter_num >= args.max_its:
            break  # 算法结束

        iter_num += 1

    return list_c

if __name__ == "__main__":
    # 数据不需要索引列
    print("请输入数据源(csv格式)的路径:")
    path = input("path: ")
    df_ = pd.read_csv(path)

    print("请输入isodata算法所需参数:")
    n_c = int(input("初始聚类中心个数: "))
    exp_k_ = int(input("希望的聚类中心数: "))
    least_n_ = int(input("聚类中最少样本数: "))
    max_its_ = int(input("迭代总次数: "))
    max_pair_ = int(input("一次迭代中允许合并的最大对数: "))
    theta_s_ = float(input("标准差阈值: "))
    theta_c_ = float(input("两聚类中心的最短距离: "))

    cluster_list = isodata(df_, n_c, exp_k_, least_n_, max_its_, max_pair_, theta_s_, theta_c_)
    for q in range(len(cluster_list)):
        print("class %d: " % q)
        for dot_index in cluster_list[q].dots:
            print(df_.loc[dot_index])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    plt.rcParams["axes.unicode_minus"] = False
    for c in range(len(cluster_list)):
        for i in cluster_list[c].dots:
            plt.plot(df_.loc[i]["x"], df_.loc[i]["y"], mark[c])
    plt.title("isodata")
    plt.show()
