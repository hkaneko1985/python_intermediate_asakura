# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 設定 ここから
window_length = 21  # 窓枠の数
polyorder = 2  # 多項式の次数
deriv = 0  # 微分次数 (0 は微分なし)
plot_spectra_number = 12  # 表示するスペクトルのサンプル番号 (0, 1, ..., 227)
# 設定 ここまで

x = pd.read_csv('sample_spectra_dataset.csv', index_col=0)  # データセットの読み込み
wavelengths = np.array(x.columns, dtype='float64')  # 波長
preprocessed_x = savgol_filter(x.values, window_length=window_length, polyorder=polyorder, deriv=deriv)  # SG 法
preprocessed_x = pd.DataFrame(preprocessed_x, index=x.index, columns=x.columns)
preprocessed_x.to_csv('preprocessed_sample_spectra_dataset_w{0}_p{1}_d{2}.csv'.format(window_length, polyorder, deriv))  # 保存

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.plot(wavelengths, x.iloc[plot_spectra_number, :], 'b-', label='original')  # プロット
plt.xlabel('wavelength [nm]')  # x 軸の名前
plt.ylabel('Absorbance')  # y 軸の名前
plt.xlim(wavelengths[0] -1, wavelengths[-1] + 1)  # x 軸の範囲の設定
plt.legend()
plt.show()  # 以上の設定で描画

plt.plot(wavelengths, preprocessed_x.iloc[plot_spectra_number, :], 'b-', label='preprocessed')  # プロット
plt.xlabel('wavelength [nm]')  # x 軸の名前
plt.ylabel('Absorbance')  # y 軸の名前
plt.xlim(wavelengths[0] -1, wavelengths[-1] + 1)  # x 軸の範囲の設定
plt.legend()
plt.show()  # 以上の設定で描画
