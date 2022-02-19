# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

# 設定 ここから
number_of_gaussians = 2  # 正規分布の数
covariance_type = 'spherical'  # 分散共分散行列の種類
# 'full' : 各正規分布がそれぞれ別の一般的な分散共分散行列をもつ
# 'tied' : すべての正規分布が同じ一般的な分散共分散行列をもつ
# 'diag' : 各正規分布がそれぞれ別の、共分散がすべて 0 の分散共分散行列をもつ
# 'spherical' : 各正規分布がそれぞれ別の、共分散がすべて 0 で分散が同じ値の分散共分散行列をもつ
# 設定 ここまで

x = pd.read_csv('sample_dataset_gmm.csv', index_col=0)  # データセットの読み込み

# オートスケーリング
autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
# プロット
plt.rcParams['font.size'] = 18
plt.scatter(autoscaled_x.iloc[:, 0], autoscaled_x.iloc[:, 1], c='blue')
plt.xlim(-4, 3)
plt.ylim(-3, 3)
plt.xlabel(x.columns[0])
plt.ylabel(x.columns[1])
plt.show()

# GMM モデリング
model = GaussianMixture(n_components=number_of_gaussians, covariance_type=covariance_type)
model.fit(autoscaled_x)

# クラスターへの割り当て
cluster_numbers = model.predict(autoscaled_x)
cluster_numbers = pd.DataFrame(cluster_numbers, index=x.index, columns=['cluster numbers'])
cluster_numbers.to_csv('cluster_numbers_gmm_{0}_{1}.csv'.format(number_of_gaussians, covariance_type))
cluster_probabilities = model.predict_proba(autoscaled_x)
cluster_probabilities = pd.DataFrame(cluster_probabilities, index=x.index)
cluster_probabilities.to_csv('cluster_probabilities_gmm_{0}_{1}.csv'.format(number_of_gaussians, covariance_type))

# プロット
x_axis = np.linspace(-4.0, 3.0)
y_axis = np.linspace(-3.0, 3.0)
X, Y = np.meshgrid(x_axis, y_axis)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model.score_samples(XX)
Z = Z.reshape(X.shape)
CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=7.1), levels=np.logspace(0, 0.85, 8)
)
plt.scatter(autoscaled_x.iloc[:, 0], autoscaled_x.iloc[:, 1], c='blue')
plt.xlim(-4, 3)
plt.ylim(-3, 3)
plt.xlabel(x.columns[0])
plt.ylabel(x.columns[1])
plt.show()
