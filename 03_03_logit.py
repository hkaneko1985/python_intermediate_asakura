# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np

values = np.arange(0.001, 1, 0.001, dtype=float)
logit = np.log(values / (1 - values))
inverse_logit = np.exp(logit) / (1 + np.exp(logit))

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.scatter(values, logit, c='blue')  # プロット
plt.xlabel('y')  # x 軸の名前
plt.ylabel('z (after logit)')  # y 軸の名前
plt.xlim(0, 1)  # x 軸の範囲の設定
plt.show()  # 以上の設定で描画

plt.figure(figsize=figure.figaspect(1))  # 図を正方形に
plt.scatter(values, inverse_logit, c='blue')  # プロット
plt.xlabel('y')  # x 軸の名前
plt.ylabel('inverse logit of z')  # y 軸の名前
plt.xlim(0, 1)  # x 軸の範囲の設定
plt.ylim(0, 1)  # y 軸の範囲の設定
plt.show()  # 以上の設定で描画
