import numpy as np
import pandas as pd

print("请输入数据源(csv格式)的路径:")
path = input("path: ")
df = pd.read_csv(path)
print("请指定索引列,输入作为数据索引的列的名称:")
index_name = input("column_name: ")
df = df.set_index([index_name])
dots = []
row_number = len(df.index)
for i in range(1, row_number+1):
    dots.append(df.loc[[i]])
columns = []
for co in df.columns:
    columns.append(co)


def calcu_distance(dot1, dot2):
    np_vec1 = np.array(dot1.values)
    np_vec2 = np.array(dot2.values)
    return np.linalg.norm(np_vec1 - np_vec2)

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
dict_ = {}
for center in centers:
    index_ = list(center.index)[0]
    dict_[index_] = []
for dot in dots:
    min_dot_center_d = np.inf
    dot_class = 0
    for center in centers:
        d = calcu_distance(dot, center)
        if d < min_dot_center_d:
            min_dot_center_d = d
            dot_class = center
    dict_[list(dot_class.index)[0]].append(dot.copy())

print(dict_)
