# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from dcekit.validation import k3nerror

# 設定 ここから
k_in_k3n_error = 10  # k3n-error の k
candidates_of_perplexity = np.arange(5, 105, 5, dtype=int)  # t-SNE の perplexity の候補
# 設定 ここまで

dataset = pd.read_csv('selected_descriptors_with_boiling_point.csv', index_col=0)  # データセットの読み込み

y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# k3n-error を用いた perplexity の最適化 
k3n_errors = []
for index, perplexity in enumerate(candidates_of_perplexity):
    print(index + 1, '/', len(candidates_of_perplexity))
    t = TSNE(perplexity=perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_x)
    scaled_t = (t - t.mean(axis=0)) / t.std(axis=0, ddof=1)

    k3n_errors.append(k3nerror(autoscaled_x, scaled_t, k_in_k3n_error) + k3nerror(scaled_t, autoscaled_x, k_in_k3n_error))
plt.scatter(candidates_of_perplexity, k3n_errors, c='blue')
plt.xlabel('perplexity')
plt.ylabel('k3n-error')
plt.show()
optimal_perplexity = candidates_of_perplexity[np.where(k3n_errors == np.min(k3n_errors))[0][0]]
print('k3n-error による perplexity の最適値 :', optimal_perplexity)

# t-SNE
t = TSNE(perplexity=optimal_perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_x)
t = pd.DataFrame(t, index=dataset.index, columns=['t_1', 't_2'])  # pandas の DataFrame 型に変換。行の名前・列の名前も設定
t.to_csv('tsne_score_perplexity_{0}.csv'.format(optimal_perplexity))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# t1 と t2 の散布図
plt.rcParams['font.size'] = 18
plt.scatter(t.iloc[:, 0], t.iloc[:, 1], c='blue')
plt.xlabel('t1')
plt.ylabel('t2')
plt.show()
