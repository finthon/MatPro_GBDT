# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
from sklearn.externals import joblib


sys.setrecursionlimit(10000000)
data = pd.read_csv(r"rutile_data.csv")
df = data[data['ang1'] != 0]
y = df['Ea'].values
x = df[['detaH', 'M4-O1', 'ang1', 'ang2', 'diangle']]

IrO2 = x.iloc[-1].values
detaH_max = x['detaH'].max()
detaH_min = x['detaH'].min()
d4_max = x['M4-O1'].max()
d4_min = x['M4-O1'].min()
a1_max = x['ang1'].max()
a1_min = x['ang1'].min()
a2_max = x['ang2'].max()
a2_min = x['ang2'].min()
da_max = x['diangle'].max()
da_min = x['diangle'].min()
print(IrO2)
# print(d4_min)
# print(d4_max)
# print(da_min)
# print(da_max)
# read
gbr = joblib.load("model.m")
# y = gbr.predict([IrO2])
# print(y)


#H-d4-da-a1-a2
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# h-d4
pred_list = []
IrO2_copy = IrO2.copy()
for i in np.linspace(detaH_min, detaH_max, 200):
    for j in np.linspace(d4_min, d4_max, 200):
        IrO2_copy[0] = i
        IrO2_copy[1] = j
        y_pred = gbr.predict([IrO2_copy])
        pred_list.append(y_pred[0])

y = np.linspace(detaH_min, detaH_max, 200)
x = np.linspace(d4_min, d4_max, 200)
X, Y = np.meshgrid(x, y)
Z = np.array(pred_list).reshape(200, 200)
a = ax1.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral.reversed())
ax1.set_xlabel('D($\AA$)', size=12)
ax1.set_ylabel('$\Delta$H(eV)', size=12)
cbar = plt.colorbar(a, ax=ax1)
cbar.set_label('$E_a$(eV)', fontsize=12)
# h-da
pred_list = []
IrO2_copy = IrO2.copy()
for i in np.linspace(detaH_min, detaH_max, 200):
    for j in np.linspace(da_min, da_max, 200):
        IrO2_copy[0] = i
        IrO2_copy[1] = j
        y_pred = gbr.predict([IrO2_copy])
        pred_list.append(y_pred[0])

y = np.linspace(detaH_min, detaH_max, 200)
x = np.linspace(da_min, da_max, 200)
X, Y = np.meshgrid(x, y)
Z = np.array(pred_list).reshape(200, 200)
a = ax2.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral.reversed())
ax2.set_xlabel('DA($^\circ)$', size=12)
ax2.set_ylabel('$\Delta$H(eV)', size=12)
cbar = plt.colorbar(a, ax=ax2)
cbar.set_label('$E_a$(eV)', fontsize=12)
# h-a1
pred_list = []
IrO2_copy = IrO2.copy()
for i in np.linspace(detaH_min, detaH_max, 200):
    for j in np.linspace(a1_min, a1_max, 200):
        IrO2_copy[0] = i
        IrO2_copy[2] = j
        y_pred = gbr.predict([IrO2_copy])
        pred_list.append(y_pred[0])

y = np.linspace(detaH_min, detaH_max, 200)
x = np.linspace(a1_min, a1_max, 200)
X, Y = np.meshgrid(x, y)
Z = np.array(pred_list).reshape(200, 200)
a = ax3.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral.reversed())
ax3.set_xlabel('A1($^\circ$)', size=12)
ax3.set_ylabel('$\Delta$H(eV)', size=12)
cbar = plt.colorbar(a, ax=ax3)
cbar.set_label('$E_a$(eV)', fontsize=12)
# h-a2
pred_list = []
IrO2_copy = IrO2.copy()
for i in np.linspace(detaH_min, detaH_max, 200):
    for j in np.linspace(a2_min, a2_max, 200):
        IrO2_copy[0] = i
        IrO2_copy[3] = j
        y_pred = gbr.predict([IrO2_copy])
        pred_list.append(y_pred[0])

y = np.linspace(detaH_min, detaH_max, 200)
x = np.linspace(a2_min, a2_max, 200)
X, Y = np.meshgrid(x, y)
Z = np.array(pred_list).reshape(200, 200)
a = ax4.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral.reversed())
ax4.set_xlabel('A2($^\circ$)', size=12)
ax4.set_ylabel('$\Delta$H(eV)', size=12)
cbar = plt.colorbar(a, ax=ax4)
cbar.set_label('$E_a$(eV)', fontsize=12)

fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.35, hspace=0.25)
plt.figtext(0.05, 0.88, 'a', fontsize=20)
plt.figtext(0.05, 0.85, 'A1=110.791$^\circ$', fontsize=9)
plt.figtext(0.05, 0.82, 'A2=69.209$^\circ$', fontsize=9)
plt.figtext(0.05, 0.79, 'DA=0$^\circ$', fontsize=9)

plt.figtext(0.51, 0.88, 'b', fontsize=20)
plt.figtext(0.51, 0.85, 'A1=110.791$^\circ$', fontsize=9)
plt.figtext(0.51, 0.82, 'A2=69.209$^\circ$', fontsize=9)
plt.figtext(0.51, 0.79, 'D=3.407$\AA$', fontsize=9)

plt.figtext(0.05, 0.435, 'c', fontsize=20)
plt.figtext(0.05, 0.405, 'D=3.407$\AA$', fontsize=9)
plt.figtext(0.05, 0.375, 'DA=0$^\circ$', fontsize=9)
plt.figtext(0.05, 0.345, 'A2=69.209$^\circ$', fontsize=9)

plt.figtext(0.51, 0.435, 'd', fontsize=20)
plt.figtext(0.51, 0.405, 'D=3.407$\AA$', fontsize=9)
plt.figtext(0.51, 0.375, 'DA=0$^\circ$', fontsize=9)
plt.figtext(0.51, 0.345, 'A1=110.791$^\circ$', fontsize=9)
plt.savefig('h-d4-da-a1-a2-603.jpg', dpi=300)
# plt.show()


# # D4-DA
# pred_list = []
# for i in np.linspace(d4_min, d4_max, 300):
#     for j in np.linspace(da_min, da_max, 300):
#         IrO2[1] = i
#         IrO2[-1] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(d4_min, d4_max, 300)
# x = np.linspace(da_min, da_max, 300)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(300, 300)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('DA($^\circ$)', size=12)
# plt.ylabel('D4($\AA$)', size=12)
# plt.colorbar(a)
# plt.show()

# # D4-A1
# pred_list = []
# for i in np.linspace(d4_min, d4_max, 200):
#     for j in np.linspace(a1_min, a1_max, 200):
#         IrO2[1] = i
#         IrO2[2] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(d4_min, d4_max, 200)
# x = np.linspace(a1_min, a1_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('A1($^\circ$)', size=12)
# plt.ylabel('D4($\AA$)', size=12)
# plt.colorbar(a)
# plt.show()


# # D4-A2
# pred_list = []
# for i in np.linspace(d4_min, d4_max, 200):
#     for j in np.linspace(a2_min, a2_max, 200):
#         IrO2[1] = i
#         IrO2[3] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(d4_min, d4_max, 200)
# x = np.linspace(a2_min, a2_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('A2($^\circ$)', size=12)
# plt.ylabel('D4($\AA$)', size=12)
# plt.colorbar(a)
# plt.show()

# # H-A1
# pred_list = []
# for i in np.linspace(detaH_min, detaH_max, 200):
#     for j in np.linspace(a1_min, a1_max, 200):
#         IrO2[0] = i
#         IrO2[2] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(detaH_min, detaH_max, 200)
# x = np.linspace(a1_min, a1_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('A1($^\circ$)', size=12)
# plt.ylabel('$\Delta$H(eV)', size=12)
# plt.colorbar(a)
# plt.show()

# # H-Da
# pred_list = []
# for i in np.linspace(detaH_min, detaH_max, 200):
#     for j in np.linspace(da_min, da_max, 200):
#         IrO2[0] = i
#         IrO2[-1] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(detaH_min, detaH_max, 200)
# x = np.linspace(da_min, da_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 20, cmap=plt.cm.Spectral)
# plt.xlabel('DA($^\circ$)', size=12)
# plt.ylabel('$\Delta$H(eV)', size=12)
# plt.colorbar(a)
# plt.show()

# # detaH-D4
# pred_list = []
# for i in np.linspace(detaH_min, detaH_max, 200):
#     for j in np.linspace(d4_min, d4_max, 200):
#         IrO2[0] = i
#         IrO2[1] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(detaH_min, detaH_max, 200)
# x = np.linspace(d4_min, d4_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('D4($\AA$)', size=12)
# plt.ylabel('$\Delta$H(eV)', size=12)
# plt.colorbar(a)
# plt.show()

# # A1-A2
# pred_list = []
# for i in np.linspace(a1_min, a1_max, 200):
#     for j in np.linspace(a2_min, a2_max, 200):
#         IrO2[2] = i
#         IrO2[3] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(a1_min, a1_max, 200)
# x = np.linspace(a2_min, a2_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('A2($^\circ$)', size=12)
# plt.ylabel('A1($^\circ$)', size=12)
# plt.colorbar(a)
# # plt.savefig('A1-A2.jpg')
# plt.show()

# # H-A2
# pred_list = []
# for i in np.linspace(detaH_min, detaH_max, 200):
#     for j in np.linspace(a2_min, a2_max, 200):
#         IrO2[0] = i
#         IrO2[3] = j
#         y_pred = gbr.predict([IrO2])
#         pred_list.append(y_pred[0])
#
# y = np.linspace(detaH_min, detaH_max, 200)
# x = np.linspace(a2_min, a2_max, 200)
# X, Y = np.meshgrid(x, y)
# Z = np.array(pred_list).reshape(200, 200)
# a = plt.contourf(X, Y, Z, 100, cmap=plt.cm.Spectral)
# plt.xlabel('A2($^\circ$)', size=12)
# plt.ylabel('$\Delta$H(eV)', size=12)
# plt.colorbar(a)
# plt.show()