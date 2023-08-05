# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict_spinel_coru
   Description :
   Author :       DrZ
   date：          2018/12/29
-------------------------------------------------
   Change Activity:
                   2018/12/29:
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

df = pd.read_csv(r'predict_ce_rh1.csv')
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

Co3O4_points_t = y[[0, 1, 4, 5, 6]]
Co3O4_points_p = y_gbr[[0, 1, 4, 5, 6]]
Rh2O3_points_t = y[[2, 3, 7, 8]]
Rh2O3_points_p = y_gbr[[2, 3, 7, 8]]
Cr2O3_points_t = y[[14, 15, 16, 17]]
Cr2O3_points_p = y_gbr[[14, 15, 16, 17]]
ZnCo2O4_points_t = y[[9, 10, 11, 12, 13]]
ZnCo2O4_points_p = y_gbr[[9, 10, 11, 12, 13]]

plt.figure(figsize=(8, 6))
plt.scatter(Co3O4_points_t, Co3O4_points_p, s=200, label='$Co_3O_4$', c='indigo', alpha=0.7)
plt.scatter(Rh2O3_points_t, Rh2O3_points_p, s=200, marker='^', label='$Rh_2O_3$', c='green', alpha=0.7)
plt.scatter(ZnCo2O4_points_t, ZnCo2O4_points_p, s=200, marker='^', label='$ZnCo_2O_3$', c='red', alpha=0.7)
plt.scatter(Cr2O3_points_t, Cr2O3_points_p, s=200, marker='^', label='$Cr_2O_3$', c='k', alpha=0.7)
plt.plot([-0.5, 1, 2, 2.5], [-0.5, 1, 2, 2.5], 'k--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$E_{DFT}$(eV)', size=16)
plt.ylabel('$E_{Predicted}$(eV)', size=16)
plt.text(1.7, 0.5, "RMSE={}".format(np.round(rr, 2)), size=14)
plt.legend(loc='best', fontsize=14)
plt.axis([-0.5, 2.5, -0.5, 2.5])
# plt.text(0.01, 0.3, "$Co_{3}O_{4}(110)$", size=12)
# plt.text(2.0, 1.7, "$Co_{3}O_{4}(100)$", size=12)
# plt.text(0.9, 0.6, "$Rh_{2}O_{3}(001)$", size=12)
# plt.text(1.5, 1.3, "$Rh_{2}O_{3}(012)$", size=12)
# plt.savefig('predict_result-603.jpg', dpi=300)
plt.show()
