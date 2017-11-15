import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kmeans
import max_min_distance_means
import hierarchical_clustering
import isodata

mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
plt.rcParams["axes.unicode_minus"] = False
df = pd.read_csv("test6.csv")

# k-means
plt.subplot(221)
belongs = kmeans.kmeans(df, 4)
for c in range(4):
    indexes = np.where(belongs == c)[0]
    for i in indexes:
        plt.plot(df.loc[i]["x"], df.loc[i]["y"], mark[c])
plt.title("k-means")

# hierarchical
plt.subplot(222)
classes = hierarchical_clustering.clustering(df, 1.7)
for c in range(len(classes)):
    for i in classes[c]:
        plt.plot(i["x"], i["y"], mark[c])
plt.title("hierarchical")

# max_min distance
plt.subplot(223)
dict_ = max_min_distance_means.max_min_dist_clustering(df)
for c in range(len(dict_)):
    for i in dict_[c]:
        plt.plot(i["x"], i["y"], mark[c])
plt.title("max_min distance")

# isodata
plt.subplot(224)
cluster_list = isodata.isodata(df, 5, 3, 3, 3, 3, 3, 3)
for c in range(len(cluster_list)):
    for i in cluster_list[c].dots:
        plt.plot(df.loc[i]["x"], df.loc[i]["y"], mark[c])
plt.title("isodata")

plt.show()
