# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict_ternaryMOs
   Description :
   Author :       DrZ
   date：          2023/8/5
-------------------------------------------------
   Change Activity:
                   2023/8/5:
-------------------------------------------------
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
from sklearn.externals import joblib


# 读取模型
gbr = joblib.load("model.m")

df = pd.read_csv(r'predict_ternaryMOs.csv')
# df = df.drop(0)
print(df)
y = df['Ea'].values
x = df.drop('Ea', axis=1)

y_gbr = gbr.predict(x)
r2 = r2_score(y, y_gbr)
mse = mean_squared_error(y, y_gbr)
rr = np.sqrt(mse)

print(y)
print(y_gbr)
print(r2)
print(rr)


plt.figure(figsize=(8, 6))
plt.scatter(y, y_gbr, s=200, c='indigo', alpha=0.7)
plt.plot([-0.5, 1, 2, 2.5], [-0.5, 1, 2, 2.5], 'k--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$E_{DFT}$(eV)', size=16)
plt.ylabel('$E_{Predicted}$(eV)', size=16)
plt.text(1.7, 0.5, "RMSE={}".format(np.round(rr, 2)), size=14)
plt.axis([0, 0.6, 0, 0.6])
plt.show()