# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

# 設定 ここから
threshold_of_rate_of_same_value = 0.8  # 同じ値をもつサンプルの割合で特徴量を削除するためのしきい値
threshold_of_r = 0.95  # 相関係数の絶対値がこの値以上となる特徴量の組は、同じクラスターになるように計算します
# 設定 ここまで

dataset = pd.read_csv('descriptors_with_logS.csv', index_col=0)  # データセットの読み込み

y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

print('最初の特徴量の数 :', x.shape[1])
# 同じ値の割合が、threshold_of_rate_of_same_value 以上の特徴量を削除
rate_of_same_value = []
for x_variable_name in x.columns:
    same_value_number = x[x_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)[0]
x = x.drop(x.columns[deleting_variable_numbers], axis=1)
print('同じ値をもつサンプルの割合で削除後の特徴量の数 :', x.shape[1])

# 相関係数に基づくクラスタリング
r_in_x = x.corr()
r_in_x = abs(r_in_x)
distance_in_x = 1 / r_in_x
for i in range(r_in_x.shape[0]):
    r_in_x.iloc[i, i] = 10 ^ 10
clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree='True',
                                     distance_threshold=1 / threshold_of_r, linkage='complete')
clustering.fit(distance_in_x)
cluster_numbers = clustering.labels_
cluster_numbers_df = pd.DataFrame(cluster_numbers, index=x.columns, columns=['cluster number'])
cluster_numbers_df.to_csv('cluster_numbers_correlation.csv')
print('相関係数に基づいてクラスタリングした後の特徴量クラスターの数: {0}'.format(cluster_numbers.max()))

# クラスターごとに一つの特徴量を選択
x_selected = pd.DataFrame([])
selected_variable_numbers = []
for i in range(cluster_numbers.max()):
    variable_numbers = np.where(cluster_numbers == i)[0]
    selected_variable_numbers.append(variable_numbers[0])
    x_selected = pd.concat([x_selected, x.iloc[:, variable_numbers[0]]], axis=1, sort=False)
x_selected.to_csv('x_selected_correlation_clustering.csv')  # 保存

deleted_variable_numbers = list(set(range(x.shape[1])) - set(selected_variable_numbers))
similarity_matrix = abs(x.corr())
similarity_matrix = similarity_matrix.iloc[selected_variable_numbers, deleted_variable_numbers]

# ヒートマップ
plt.rcParams['font.size'] = 12
sns.heatmap(similarity_matrix, vmax=1, vmin=0, cmap='seismic', xticklabels=1, yticklabels=1)
plt.xlim([0, similarity_matrix.shape[1]])
plt.ylim([0, similarity_matrix.shape[0]])
plt.show()

similarity_matrix.to_csv('similarity_matrix_correlation_clustering.csv')  # 保存

# クラスターごとに特徴量を平均化
x_averaged = pd.DataFrame([])
for i in range(cluster_numbers.max()):
    variable_numbers = np.where(cluster_numbers == i)[0]
    if variable_numbers.shape[0] == 1:
        x_averaged = pd.concat([x_averaged, x.iloc[:, variable_numbers[0]]], axis=1, sort=False)
    else:
        x_each_cluster = x.iloc[:, variable_numbers]
        autoscaled_x_each_cluster = (x_each_cluster - x_each_cluster.mean()) / x_each_cluster.std()
        averaged_x_each_cluster = autoscaled_x_each_cluster.mean(axis=1)
        averaged_x_each_cluster.name = 'mean_in_cluster_{0}'.format(i)
        x_averaged = pd.concat([x_averaged, averaged_x_each_cluster], axis=1, sort=False)

x_selected.to_csv('x_averaged_correlation_clustering.csv')  # 保存
