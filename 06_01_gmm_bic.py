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
numbers_of_gaussians = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 正規分布の数の候補
covariance_types = ['full', 'tied', 'diag', 'spherical']  # 分散共分散行列の種類の候補
# 'full': 各正規分布がそれぞれ別の一般的な分散共分散行列をもつ
# 'tied': すべての正規分布が同じ一般的な分散共分散行列をもつ
# 'diag': 各正規分布がそれぞれ別の、共分散がすべて 0 の分散共分散行列をもつ
# 'spherical': 各正規分布がそれぞれ別の、共分散がすべて 0 で分散が同じ値の分散共分散行列をもつ
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

# BIC によるクラスタ数と分散共分散行列の最適化
bics = np.zeros([len(covariance_types), len(numbers_of_gaussians)])
for index_cov, covariance_type in enumerate(covariance_types):
    for index_num, number_of_gaussians in enumerate(numbers_of_gaussians):
        model = GaussianMixture(n_components=number_of_gaussians, covariance_type=covariance_type)
        model.fit(autoscaled_x)
        bics[index_cov, index_num] = model.bic(autoscaled_x)

colors = ['blue', 'red', 'green', 'black']
for i in range(len(covariance_types)):
    plt.scatter(numbers_of_gaussians, bics[i, :], c=colors[i], label=covariance_types[i])
plt.xlabel('number of Gaussians')
plt.ylabel('BIC')
plt.legend()
plt.show()

# BIC が最小となる正規分布の数と分散共分散行列の種類
opt_cov_index, opt_num_index = np.where(bics == bics.min())
optimal_number_of_gaussians = numbers_of_gaussians[opt_num_index[0]]
optimal_covariance_type = covariance_types[opt_cov_index[0]]
print('最適化された正規分布の数 :', optimal_number_of_gaussians)
print('最適化された分散共分散行列の種類 :', optimal_covariance_type)

# GMM モデリング
model = GaussianMixture(n_components=optimal_number_of_gaussians, covariance_type=optimal_covariance_type)
model.fit(autoscaled_x)

# クラスターへの割り当て
cluster_numbers = model.predict(autoscaled_x)
cluster_numbers = pd.DataFrame(cluster_numbers, index=x.index, columns=['cluster numbers'])
cluster_numbers.to_csv('cluster_numbers_gmm_{0}_{1}.csv'.format(optimal_number_of_gaussians, optimal_covariance_type))
cluster_probabilities = model.predict_proba(autoscaled_x)
cluster_probabilities = pd.DataFrame(cluster_probabilities, index=x.index)
cluster_probabilities.to_csv('cluster_probabilities_gmm_{0}_{1}.csv'.format(optimal_number_of_gaussians, optimal_covariance_type))

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
