# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from SearchTetrahedron import SearchTetrahedron


class GBDTML:
    """
    According to site1, site2, site3 and site4 generated D, DA, A1, A2; with a reasonable deltaH to
    predict the activation energy Ea
    """
    def __init__(self, model):
        """
        read the model
        """
        self.gbr = joblib.load(model)

    def predict_Ea(self,  D, A1, A2, DA, detaH=-1.2):
        """
        Given x, output the predcit y
        """
        input_x = np.array([[detaH, D, A1, A2, DA]])
        y_pred = self.gbr.predict(input_x)
        return y_pred

    def save_data(self):
        cpath = os.getcwd()
        bulk_dir = os.path.join(cpath, 'bulk_database')
        num = 1
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            num = len(df) + 1
        else:
            df = pd.DataFrame(
                columns=['id', 'Ea', 'D', 'A1', 'A2', 'DA', 'd4', 'd3', 'point1', 'point2', 'point3', 'point4'])
        for no, i in enumerate(os.listdir(bulk_dir)):
            # if i != 'POSCAR_mp-1181286':
            #     continue
            mat_id = i.split('_')[-1]
            if mat_id in df['id'].values:
                continue
            print('=== searching {} structure ==='.format(no+1))
            print(i)
            abpath = os.path.join(bulk_dir, i)
            st = SearchTetrahedron(abpath)
            genes_and_point = st.search_rule()
            for j in genes_and_point:
                genes = j[0]  # genes list [D, A1, A2, DA, d4, d3]
                points = j[1]  # points list [point1, point2, point3, point4]
                d = genes[0]
                a1 = genes[1]
                a2 = genes[2]
                da = genes[3]
                d4 = genes[4]
                d3 = genes[5]
                point1 = points[0]
                point2 = points[1]
                point3 = points[2]
                point4 = points[3]
                y_pred = self.predict_Ea(d, a1, a2, da)
                df1 = pd.DataFrame([[st.file_id, y_pred[0], d, a1, a2, da, d4, d3, point1, point2, point3, point4]],
                                   columns=['id', 'Ea', 'D', 'A1', 'A2', 'DA', 'd4', 'd3', 'point1', 'point2', 'point3',
                                            'point4'],
                                   index=[num])
                df = df.append(df1)
                print('{} done.'.format(num))
                num += 1
            print('===  {} structure finished ==='.format(no+1))
            df.to_csv('data.csv', index=None)


if __name__ == '__main__':
    model_dir = r'model.m'
    gb = GBDTML(model_dir)
    gb.save_data()  # get data.csv
    # abpath = r'./bulk_database/POSCAR_mp-2723'

    # df = pd.read_csv('data.csv')
    # print(df)
    # df1 = df[df['Ea'] < 0.1]
    # print(df1)
    # print(df['point1'])
    # for i in df['point1']:
    #     print(i)
    #     print(type(i))
    #     i = i.strip('[]')
    #     print(re.split('\s+', i))
    #     input()







