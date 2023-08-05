# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def K_fold_cross_val(X, Y, cv=4, circle=1):
    """
    K-fold cross-validation for origin data, some wrong answers can be got from scikit-learn, so I rewrite this function
    """
    min_rr = 10
    size = len(Y)
    circle_list = []
    for cir in range(circle):
        score_list = []
        
        n_array = np.random.permutation(size).reshape(cv, -1)
        for i in range(cv):
            train_index = n_array[np.array([ii for ii in range(cv) if ii != i])].flatten()
            test_index = n_array[i, :]
            x_test = X[test_index]
            x_train = X[train_index]
            y_train =Y[train_index]
            y_test = Y[test_index]
            gbr = GradientBoostingRegressor(n_estimators=150, max_depth=2, min_samples_split=3, learning_rate=0.2,
                                            loss='ls')
            gbr = gbr.fit(x_train, y_train.ravel())
            y_gbr1 = gbr.predict(x_test)
            mse = mean_squared_error(y_test, y_gbr1)
            rr = np.sqrt(mse)
            if rr < min_rr:
                min_rr = rr
            score_list.append(rr)
        print(np.mean(score_list))
        circle_list.append(np.mean(score_list))
    print('{}circle，{} RMSE={}'.format(circle, cv, np.mean(circle_list)))
    return np.mean(circle_list)


sys.setrecursionlimit(10000000)
data = pd.read_csv(r"rutile_data.csv")
df = data[data['ang1'] != 0]
y = df['Ea'].values
x = df[['detaH', 'M4-O1', 'ang1', 'ang2', 'diangle']].values

K_fold_cross_val(x, y, cv=4, circle=2000)