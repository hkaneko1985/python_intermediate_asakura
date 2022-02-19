# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 設定 ここから
threshold_of_rate_of_same_value = 0.8  # 同じ値をもつサンプルの割合で特徴量を削除するためのしきい値
threshold_of_r = 0.95  # 相関係数の絶対値がこの値以上となる特徴量の組の一方を削除します
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

# 相関係数の絶対値が threshold_of_r 以上となる特徴量の組の一方を削除
r_in_x = x.corr()
r_in_x = abs(r_in_x)
for i in range(r_in_x.shape[0]):
    r_in_x.iloc[i, i] = 0
deleted_variable_numbers = []
for i in range(r_in_x.shape[0]):
    r_max = r_in_x.max()
    r_max_max = r_max.max()
    if r_max_max >= threshold_of_r:
#        print(i + 1)
        variable_number_1 = np.where(r_max == r_max_max)[0][0]
        variable_number_2 = np.where(r_in_x.iloc[:, variable_number_1] == r_max_max)[0][0]
        r_sum_1 = r_in_x.iloc[:, variable_number_1].sum()
        r_sum_2 = r_in_x.iloc[:, variable_number_2].sum()
        if r_sum_1 >= r_sum_2:
            delete_x_number = variable_number_1
        else:
            delete_x_number = variable_number_2
        deleted_variable_numbers.append(delete_x_number)
        r_in_x.iloc[:, delete_x_number] = 0
        r_in_x.iloc[delete_x_number, :] = 0
    else:
        break

x_selected = x.drop(x.columns[deleted_variable_numbers], axis=1)
print('相関係数で削除後の特徴量の数 :', x_selected.shape[1])

x_selected.to_csv('x_selected_correlation_deleting.csv')  # 保存

selected_variable_numbers = list(set(range(x.shape[1])) - set(deleted_variable_numbers))
similarity_matrix = abs(x.corr())
similarity_matrix = similarity_matrix.iloc[selected_variable_numbers, deleted_variable_numbers]

# ヒートマップ
plt.rcParams['font.size'] = 12
sns.heatmap(similarity_matrix, vmax=1, vmin=0, cmap='seismic', xticklabels=1, yticklabels=1)
plt.xlim([0, similarity_matrix.shape[1]])
plt.ylim([0, similarity_matrix.shape[0]])
plt.show()

similarity_matrix.to_csv('similarity_matrix_correlation_deleting.csv')  # 保存
