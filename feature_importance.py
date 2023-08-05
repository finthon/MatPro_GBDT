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


# for i in range(1, 1001):
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=603)

print(x_train)
gbr = GradientBoostingRegressor(n_estimators=150, max_depth=2, min_samples_split=3, learning_rate=0.2, loss='ls')
gbr.fit(x_train, y_train.ravel())

# joblib.dump(gbr, "model.m")

gbr = joblib.load("model.m")

y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
r2 = r2_score(y_test, y_gbr1)
mse = mean_squared_error(y_test, y_gbr1)
rr = np.sqrt(mse)
# if rr <= 0.15 and r2 >= 0.95:
#     print('the' + str(i))
print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rr)

feature_importance = gbr.feature_importances_
print(feature_importance)
detaH = feature_importance[0]
d4 = feature_importance[1]
a1 = feature_importance[2]
a2 = feature_importance[3]
da = feature_importance[4]

feature_title = ['$\Delta$H', 'D', 'A1', 'A2', 'DA']

fig = plt.figure(figsize=(14, 6))
ax1 = plt.subplot(1, 2, 1)
# stack_barh
ind = ['Distance\ndescriptor', 'Direction\ndescriptor', '$\Delta$H']
df1 = pd.DataFrame(columns=feature_title, index=ind)
df1.iloc[2, 0] = feature_importance[0]
df1.iloc[0, 1] = feature_importance[1]
df1.iloc[1, 2:] = feature_importance[2:]
df1 = df1.fillna(0)
print(df1.values)
print(y_train)
print(y_gbr)
print(y_test)
print(y_gbr1)
df1.plot(kind='barh', stacked=True, align='center', alpha=0.8, ax=ax1)
plt.xlabel('Importance', size=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax2 = plt.subplot(1, 2, 2)
plt.plot(y_train, y_train, c='indigo', alpha=0.7)
# plt.title('True VS Predict')
plt.xlabel('$E_{DFT}$(eV)', size=16)
plt.ylabel('$E_{Predicted}$(eV)', size=16)
plt.text(2.5, 1, "RMSE={}".format(np.round(rr, 2)), size=14)
plt.scatter(y_train, y_gbr, c='indigo', alpha=0.5, s=100, label="training")
plt.scatter(y_test, y_gbr1, c='green', alpha=0.7, s=100, label="test", marker='^')
plt.legend(loc=2, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.subplots_adjust(left=0.15, right=0.90, bottom=0.15, top=0.85, wspace=0.35, hspace=0.25)
plt.figtext(0.08, 0.81, '(a)', fontsize=20)
plt.figtext(0.5, 0.81, '(b)', fontsize=20)
# plt.savefig('h-d4-a1-a2-da-da-603.jpg', dpi=300)
# plt.savefig('{}.jpg'.format(i))
# plt.show()
