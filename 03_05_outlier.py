# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# 設定 ここから
type_of_samples = 0  # 仮想サンプルの種類 0:正規乱数、1:時系列
window_length = 61  # SG 法における窓枠の数
polyorder = 2  # SG 法における多項式の次数
# 設定 ここまで

deriv = 0  # SG 法における微分次数 (0 は微分なし)

number_of_samples = 300  # 仮想サンプルの数
noise_rate = 8  # SN比
np.random.seed(10)

if type_of_samples == 0:
    outliers = [-20, 6, -6, 25]  # 外れ値
    outlier_indexes = [100, 150, 200, 250]  # 外れ値のインデックス
    x = np.random.randn(number_of_samples)
elif type_of_samples == 1:
    outliers = [1, 3, 10, -2]  # %外れ値
    outlier_indexes = [80, 150, 200, 250]  # 外れ値のインデックス
    x = np.sin(np.arange(number_of_samples) * np.pi / 50)
    noise = np.random.randn(number_of_samples)
    noise = noise * (x.var() / noise_rate) ** 0.5
    x += noise
x[outlier_indexes] = outliers  # 外れ値の追加

# 生成した仮想サンプルのプロット
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.plot(x, 'b.')  # プロット
plt.plot(outlier_indexes, x[outlier_indexes], 'r.', label='original outliers')  # プロット
plt.xlabel('sample number')  # x 軸の名前
plt.ylabel('x or y')  # y 軸の名前
plt.legend()
plt.show()  # 以上の設定で描画

# 3 sigma method
upper_3_sigma = x.mean() + 3 * x.std()
lower_3_sigma = x.mean() - 3 * x.std()
plt.plot(x, 'b.') 
plt.plot(outlier_indexes, x[outlier_indexes], 'r.', label='original outliers')
plt.plot([0, len(x)], [upper_3_sigma, upper_3_sigma], 'k-')
plt.plot([0, len(x)], [lower_3_sigma, lower_3_sigma], 'k-')
plt.xlabel('sample number')  # x 軸の名前
plt.ylabel('x or y')  # y 軸の名前
plt.title('3 sigma method')
plt.legend()
plt.show()  # 以上の設定で描画

# Hampel identifier
upper_hampel = np.median(x) + 3 * 1.4826 * np.median(np.absolute(x - np.median(x)))
lower_hampel = np.median(x) - 3 * 1.4826 * np.median(np.absolute(x - np.median(x)))
plt.plot(x, 'b.') 
plt.plot(outlier_indexes, x[outlier_indexes], 'r.', label='original outliers')
plt.plot([0, len(x)], [upper_hampel, upper_hampel], 'k-')
plt.plot([0, len(x)], [lower_hampel, lower_hampel], 'k-')
plt.xlabel('sample number')  # x 軸の名前
plt.ylabel('x or y')  # y 軸の名前
plt.title('Hampel identifier')
plt.legend()
plt.show()  # 以上の設定で描画

# SG method + Hampel identifier
preprocessed_x = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=deriv)  # SG 法
x_diff = x - preprocessed_x
upper_sg_hampel = preprocessed_x + np.median(x_diff) + 3 * 1.4826 * np.median(np.absolute(x_diff - np.median(x_diff)))
lower_sg_hampel = preprocessed_x + np.median(x_diff) - 3 * 1.4826 * np.median(np.absolute(x_diff - np.median(x_diff)))
plt.plot(x, 'b.') 
plt.plot(outlier_indexes, x[outlier_indexes], 'r.', label='original outliers')
plt.plot(range(len(x)), upper_sg_hampel, 'k-')
plt.plot(range(len(x)), lower_sg_hampel, 'k-')
plt.xlabel('sample number')  # x 軸の名前
plt.ylabel('x or y')  # y 軸の名前
plt.title('SG method + Hampel identifier')
plt.legend()
plt.show()  # 以上の設定で描画
