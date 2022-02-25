# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of sparce generative topographic mapping
# (SGTM) https://pubs.acs.org/doi/10.1021/acs.jcim.8b00528

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcekit.generative_model import GTM

# 設定 ここから
shape_of_map = [30, 30]  # ２次元平面上のグリッド点の数 (k × k)
shape_of_rbf_centers = [5, 5]  # RBF の数 (q × q)
variance_of_rbfs = 4  # RBF の分散 (σ^2)
lambda_in_em_algorithm = 0.001  # 正則化項 (λ)
number_of_iterations = 300  # EM アルゴリズムにおける繰り返し回数
display_flag = True  # EM アルゴリズムにおける進捗を表示する (True) かしない (Flase) か
# 設定 ここまで

# load dataset
dataset = pd.read_csv('selected_descriptors_with_boiling_point.csv', index_col=0)  # データセットの読み込み

y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# construct SGTM model
model = GTM(shape_of_map, shape_of_rbf_centers, variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations,
            display_flag, sparse_flag=True)
model.fit(autoscaled_x)

if model.success_flag:
    # calculate responsibilities
    responsibilities = model.responsibility(autoscaled_x)
    means, modes = model.means_modes(autoscaled_x)
    
    means_pd = pd.DataFrame(means, index=x.index, columns=['t1 (mean)', 't2 (mean)'])
    modes_pd = pd.DataFrame(modes, index=x.index, columns=['t1 (mode)', 't2 (mode)'])
    means_pd.to_csv('sgtm_means_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(
            shape_of_map[0], shape_of_map[1],
            shape_of_rbf_centers[0], shape_of_rbf_centers[1],
            variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations))
    modes_pd.to_csv('sgtm_modes_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(
            shape_of_map[0], shape_of_map[1],
            shape_of_rbf_centers[0], shape_of_rbf_centers[1],
            variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations))
    
    plt.rcParams['font.size'] = 18
    # plot the mean of responsibilities
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(means[:, 0], means[:, 1])
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('t1 (mean)')
    plt.ylabel('t2 (mean)')
    ax.set_aspect('equal')
    plt.show()
    
    # plot the mean of responsibilities
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if y.dtype == 'O':
            plt.scatter(means[:, 0], means[:, 1], c=pd.factorize(y)[0])
        else:
            plt.scatter(means[:, 0], means[:, 1], c=y)
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('t1 (mean)')
        plt.ylabel('t2 (mean)')
        plt.colorbar()
        ax.set_aspect('equal')
        plt.show()
    except NameError:
        print('y がありません')

    # plot the mode of responsibilities
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(modes[:, 0], modes[:, 1])
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('t1 (mode)')
    plt.ylabel('t2 (mode)')
    ax.set_aspect('equal')
    plt.show()
    
    # plot the mode of responsibilities
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if y.dtype == 'O':
            plt.scatter(modes[:, 0], modes[:, 1], c=pd.factorize(y)[0])
        else:
            plt.scatter(modes[:, 0], modes[:, 1], c=y)
        plt.ylim(-1.1, 1.1)
        plt.xlim(-1.1, 1.1)
        plt.xlabel('t1 (mode)')
        plt.ylabel('t2 (mode)')
        plt.colorbar()
        ax.set_aspect('equal')
        plt.show()
    except NameError:
        print('y がありません')

    # integration of clusters based on Bayesian information criterion (BIC)
    clustering_results = np.empty([3, responsibilities.shape[1]])
    cluster_numbers = np.empty(responsibilities.shape)
    original_mixing_coefficients = model.mixing_coefficients.copy()
    for i in range(len(original_mixing_coefficients)):
        likelihood = model.likelihood(autoscaled_x)
        non0_indexes = np.where(model.mixing_coefficients != 0)[0]
    
        responsibilities = model.responsibility(autoscaled_x)
        cluster_numbers[:, i] = responsibilities.argmax(axis=1)
    
        bic = -2 * likelihood + len(non0_indexes) * np.log(autoscaled_x.shape[0])
        clustering_results[:, i] = np.array([len(non0_indexes), bic, len(np.unique(responsibilities.argmax(axis=1)))])
    
        if len(non0_indexes) == 1:
            break
    
        non0_mixing_coefficient = model.mixing_coefficients[non0_indexes]
        model.mixing_coefficients[non0_indexes[non0_mixing_coefficient.argmin()]] = 0
        non0_indexes = np.delete(non0_indexes, non0_mixing_coefficient.argmin())
        model.mixing_coefficients[non0_indexes] = model.mixing_coefficients[non0_indexes] + min(
            non0_mixing_coefficient) / len(non0_indexes)
    
    clustering_results = np.delete(clustering_results, np.arange(i, responsibilities.shape[1]), axis=1)
    cluster_numbers = np.delete(cluster_numbers, np.arange(i, responsibilities.shape[1]), axis=1)
    
    plt.scatter(clustering_results[0, :], clustering_results[1, :], c='blue')
    plt.xlabel('number of clusters')
    plt.ylabel('BIC')
    plt.show()
    
    number = np.where(clustering_results[1, :] == min(clustering_results[1, :]))[0][0]
    print('最適化されたクラスター数 : {0}'.format(int(clustering_results[0, number])))
    clusters = cluster_numbers[:, number].astype('int64')
    clusters = pd.DataFrame(clusters, index=x.index, columns=['cluster numbers'])
    clusters.to_csv('cluster_numbers_sgtm_{0}_{1}_{2}_{3}_{4}_{5}_{6}.csv'.format(
            shape_of_map[0], shape_of_map[1],
            shape_of_rbf_centers[0], shape_of_rbf_centers[1],
            variance_of_rbfs, lambda_in_em_algorithm, number_of_iterations))
    
    # plot the mean of responsibilities with cluster information
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(means[:, 0], means[:, 1], c=clusters.iloc[:, 0])
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('t1 (mean)')
    plt.ylabel('t2 (mean)')
    ax.set_aspect('equal')
    plt.show()

    # plot the mode of responsibilities with cluster information
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(modes[:, 0], modes[:, 1], c=clusters.iloc[:, 0])
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.xlabel('t1 (mode)')
    plt.ylabel('t2 (mode)')
    ax.set_aspect('equal')
    plt.show()
    
