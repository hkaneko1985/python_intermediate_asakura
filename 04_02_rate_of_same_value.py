# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd

# 設定 ここから
threshold_of_rate_of_same_value = 0.8  # 同じ値をもつサンプルの割合で特徴量を削除するためのしきい値
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
x_selected = x.drop(x.columns[deleting_variable_numbers], axis=1)
print('削除後の特徴量の数 :', x_selected.shape[1])

x_selected.to_csv('x_selected_same_value.csv')  # 保存
