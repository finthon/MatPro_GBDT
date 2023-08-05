# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     search_rule
   Description :
   Author :       DrZ
   date：          2020/9/9
-------------------------------------------------
   Change Activity:
                   2020/9/9:
-------------------------------------------------
"""
import os
import re
import numpy as np
from math import acos, degrees
from vaspy.atomco import PosCar
from itertools import combinations
from vaspy.matstudio import XsdFile


class SearchTetrahedron:
    """
    searching rule, including：
    1. Calculating the lattice parameters according to the lattice matrix（a, b, c, alpha, beta, gamma）
    2. Calculating the distance between two coordinations
    3. Calculating the angle between three coordinations
    4. Calculating the diangle between four coordinations
    5. Calculating the coord and dist matrix for primitive cell and extended cell
    6. Identifying the point 1,2,3,4 of tetrahedron
    ps: The distribution of points1,2,3,4 in space is 4-1-2-3, where 1-2 can be regarded as two atoms on the surface
        atom1 and atom2; and 4 and 3 are adsorbate fragment molecules
    """
    def __init__(self, struct_file):
        """
        struct_file: POSCAR, CONTCAR or xsd
        dis_c: the cutoff of distance
        dangle_c: the cutoff of diangle
        """
        self.struct_file = struct_file
        # read POSCAR, CONTCAR or xx.xsd file
        file_type = os.path.basename(self.struct_file)
        self.file_id = file_type.split('_')[-1]
        if re.search('POSCAR', file_type):
            self.pos = PosCar(self.struct_file)
        elif re.search('CONTCAR', file_type):
            self.pos = PosCar(self.struct_file)
        elif file_type.split('.')[-1] == 'xsd':
            self.pos = XsdFile(self.struct_file)
        else:
            raise Exception('the name of struct_file must include "POSCAR" or "CONTCAR", or ".xsd" type.')

    def cal_lattice_lengths(self, bases):
        """
        input a lattice matrix, return lattice abc
        """
        return np.sqrt(np.sum(bases ** 2, axis=1)).tolist()

    def cal_lattice_angles(self, bases):
        """
        input a lattice matrix, return alpha, beta, gamma
        """
        angle = lambda X, Y: degrees(acos(np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))))
        gamma, beta, alpha = [angle(X, Y) for X, Y in combinations(bases, 2)]
        return [alpha, beta, gamma]

    def dist(self, site1, site2):
        """
        return the distance between two coordinations
        """
        return np.linalg.norm(site1 - site2)

    def angle(self, site1, site2, site3):
        """
        return the angle between of three coordinations
        """
        m1 = site1 - site2
        m2 = site3 - site2
        m = m1.dot(m2) / (np.sqrt(m1.dot(m1)) * np.sqrt(m2.dot(m2)))
        if m > 1:
            m = 1
        elif m < -1:
            m = -1
        arc = np.arccos(m)
        return arc * 180 / np.pi

    def di_angle(self, site1, site2, site3, site4):
        """
        return the diangle between plane formed by site1, site2, site4 and plane formed by site1, site2, site3
        """
        m1 = site4 - site1
        m2 = site2 - site1
        # Plane normal vector
        n1 = np.cross(m1, m2)
        m3 = site1 - site2
        m4 = site3 - site2
        n2 = np.cross(m3, m4)
        m = n1.dot(n2) / (np.sqrt(n1.dot(n1)) * np.sqrt(n2.dot(n2)))
        if m > 1:
            m = 1
        elif m < -1:
            m = -1
        arc = np.arccos(m)
        if arc - 0 < 10E-5:
            return arc * 180 / np.pi
        # the normal vector of two normal vector dot site1-2 vector
        # if plane formed by site1,2,3 is outside plane by site1,2,4, the diangle is positive
        zc = np.cross(n1, n2)
        if np.dot(zc, m2) < 0:
            return -arc * 180 / np.pi
        else:
            return arc * 180 / np.pi

    def get_all_distance_and_coord(self):
        """
        return the dis_array and coord_array of all atoms of a cell.
        the size is: [total_atoms, total_atoms, 27], which includes the distance between themselves,
        considering periodic conditions.
        There are 27 dis between two atoms, thereinto:
        1. at the primitive-primitive cell, one dis
        2. at the primitive-extended cell, 26 dis
        the point1 is selected in primitive cell, all the distance in dis_array is referenced by the atoms in the
        primitive cell. for example, the dis 2.5 indicates that one of the coord is in the primitive cell, the other
        atom could in the primitive cell or extended cell.
        """
        # lattice paramaters
        aa = self.pos.bases[0]
        bb = self.pos.bases[1]
        cc = self.pos.bases[2]
        # convert to cartesian
        cart = self.pos.dir2cart(self.pos.bases, self.pos.data)
        if cart.shape == (3,):
            cart = cart.reshape(-1, 3)
        # all dis and coord array
        dis_array = np.zeros((len(cart), len(cart), 27))
        coord_array = np.zeros((len(cart), len(cart), 27, 3))
        # traverse all the atoms in primitive cell, as the point_one
        for num, point_one in enumerate(cart):
            for num1, point in enumerate(cart):
                tran1 = point + aa
                tran2 = point + aa + bb
                tran3 = point + aa - bb
                tran4 = point + aa + cc
                tran5 = point + aa - cc
                tran6 = point + aa + bb + cc
                tran7 = point + aa + bb - cc
                tran8 = point + aa - bb + cc
                tran9 = point + aa - bb - cc
                tran10 = point - aa
                tran11 = point - aa + bb
                tran12 = point - aa - bb
                tran13 = point - aa + cc
                tran14 = point - aa - cc
                tran15 = point - aa + bb + cc
                tran16 = point - aa + bb - cc
                tran17 = point - aa - bb + cc
                tran18 = point - aa - bb - cc
                tran19 = point + bb
                tran20 = point - bb
                tran21 = point + cc
                tran22 = point - cc
                tran23 = point + bb + cc
                tran24 = point + bb - cc
                tran25 = point - bb + cc
                tran26 = point - bb - cc
                point_sum = [point, tran1, tran2, tran3, tran4, tran5, tran6, tran7, tran8, tran9, tran10,
                            tran11, tran12, tran13, tran14, tran15, tran16, tran17, tran18, tran19, tran20,
                            tran21, tran22, tran23, tran24, tran25, tran26]
                # calculating the dis corresponding to point1
                dis_sum = []
                for wf in range(27):
                    dis_sum.append(self.dist(point_one, point_sum[wf]))
                dis_array[num, num1, :] = dis_sum
                coord_array[num, num1, :] = point_sum

        return dis_array, coord_array

    def cal_bonding_criterion(self, dis_array):
        """
        checking all first-neighboring dis of all atoms of the cell, and take the longest one as the bonding
        criterion, for evaluating bond1-4 and bond2-3
        """
        bond_list = []
        for i in range(len(dis_array)):
            cc = dis_array[i].reshape(1, -1)
            cc = np.sort(cc)[0, 1:]
            for idx, j in enumerate(cc):
                n = cc[0: idx + 1]
                std_n = np.std(n)
                n1 = cc[0: idx + 2]
                std_n1 = np.std(n1)
                # after adding the next-neighboring atom, the std large than 0.2
                if std_n1 - std_n >= 0.13:
                    bond_list.append(j)
                    break
        # bond_list: [2.1090327694212734, 2.1090327694212734, 1.9267619309710815, 2.071820963032364, 39.28373337893162,
        # 1.926761930971081, 2.0718209630323643, 39.283733378931615]
        # some trouble here, recalculate the bonding_c
        if np.std(bond_list) > 1:
            bond_list.sort()
            final_list = []
            for idxx, jj in enumerate(bond_list):
                aa = bond_list[0: idxx + 1]
                std_aa = np.std(aa)
                bb = bond_list[0: idxx + 2]
                std_bb = np.std(bb)
                # after adding the next-neighboring atom, the std large than 0.2
                if std_bb - std_aa >= 0.13:
                    final_list.append(jj)
                    break
            return np.max(final_list) + 0.1

        else:
            return np.max(bond_list) + 0.1

    def search_rule(self):
        """
        the distribution of point1,2,3,4 in space is 4-1-2-3, of which 1-2 can be regarded as surface atom1 and atom2,
        whereas atom3 and atom4 are the fragment molecules, the searching rule is:
            1. the dis between point1 and point2 is less than the cutoff of dis, default 4
            2. the dis between point1 and point4 (d4) is less than the bonding dis (calculated from bulk), bonding_c
            3. the dis between point2 and point3 (d3) is less than the bonding dis
            4. 75 <= A1(point4,1,2) <= 135
            5. A2(point3,2,1) < 70
            6. the absolute value of DA[surf(4,1,2), surf(3,2,1)] <= 50
        --------------------------------------------------------------------------
        return: the list including genes and points [genes, points]
        """
        points_list = []  # save point1,2,3,4
        param_list = []  # save D, A1, A2, DA, d4, d3
        dis_array, coord_array = self.get_all_distance_and_coord()
        # get bonding criterion
        bonding_c = self.cal_bonding_criterion(dis_array)
        for idx1 in range(len(dis_array)):      # idx1 is the index of the atom in the primitive cell
            for idx2 in range(len(dis_array)):
                for idx3 in range(27):          # After considering periodicity, the search space consists of 27 cells
                    if (dis_array[idx1, idx2, idx3] > 0) and (dis_array[idx1, idx2, idx3] <= 4):
                        # searching point1 and point2
                        point1 = coord_array[idx1, idx1, 0]
                        # The clone idx3 of the idx2 atom related to the idx1 atom
                        point2 = coord_array[idx1, idx2, idx3]
                        # calculating D
                        D = np.round(dis_array[idx1, idx2, idx3], 3)
                        # searching point4
                        dis_point1_array = dis_array[idx1]
                        coord_point1_array = coord_array[idx1]
                        reshape_p1_array = dis_point1_array.reshape(1, -1)
                        # point1 as start point, get the sorted dis from small to large for searching point4
                        idx1_sort = np.argsort(reshape_p1_array)[0, :]
                        for n in idx1_sort:
                            x = n // 27     # The divisor represents the row
                            y = n % 27      # The remainder represents the column
                            # exclude point1和point2
                            if ((x == idx2) and (y == idx3)) or ((x == idx1) and (y == 0)):
                                continue
                            else:
                                point4 = coord_point1_array[x, y]
                                # exclude point4,1,2 collinear
                                tom = np.cross(point4-point1, point2-point1)
                                if len(np.unique(tom)) == 1 and np.unique(tom)[0] == 0:
                                    continue
                                # exclude zero angle of 4,1,2
                                if self.angle(point4, point1, point2) - 0 < 10E-5:
                                    continue
                                # exclude too far dis between atom1,4
                                # d4 is the distance between point1 and point4
                                d4 = np.round(dis_point1_array[x, y], 3)
                                if d4 > bonding_c:
                                    continue
                                # searching point3
                                # taking point2 as the dis array
                                dis_point2_array = dis_array[idx2]
                                # note that coord_array[idx2] is same with coord_array[idx1]
                                coord_point2_array = coord_array[idx2]
                                reshape_p2_array = dis_point2_array.reshape(1, -1)
                                # Different from finding point4, here is based on the point of point2 in the original
                                # unit cell as the starting point, and all distances are sorted from small to large
                                idx2_sort = np.argsort(reshape_p2_array)[0, :]
                                for nn in idx2_sort:
                                    xx = nn // 27
                                    yy = nn % 27
                                    # exclude point1, point2 and point4
                                    if ((xx == idx1) and (yy == 0)) or ((xx == idx2) and (yy == idx3)) \
                                            or ((xx == x) and (yy == y)):
                                        continue
                                    else:
                                        point3 = coord_point2_array[xx, yy]
                                        # exclude point3,2,1 collinear
                                        cow = np.cross(point1 - point2, point3 - point2)
                                        if len(np.unique(cow)) == 1 and np.unique(cow)[0] == 0:
                                            continue
                                        # d3 is the distance between point2 and point3
                                        d3 = np.round(self.dist(point2, point3), 3)
                                        if self.angle(point3, point2, point1) - 0 < 10E-5:
                                            continue
                                        if d3 > bonding_c:
                                            continue
                                        # calculating A1,A2,DA
                                        A1 = np.round(self.angle(point4, point1, point2), 3)
                                        A2 = np.round(self.angle(point3, point2, point1), 3)
                                        DA = np.round(self.di_angle(point1, point2, point3, point4), 3)
                                        if (75 <= A1 <= 135) and (A2 < 70) and (abs(DA) <= 50):
                                            if [D, A1, A2, DA, d4, d3] not in param_list:
                                                param_list.append([D, A1, A2, DA, d4, d3])
                                                points_list.append([point1, point2, point3, point4])

        genes_and_points = [[param_list[i], points_list[i]] for i in range(len(param_list))]
        return genes_and_points


if __name__ == '__main__':
    pwd = r'./bulk_database/POSCAR_mp-1288'
    # pwd = r'./bulk_database/POSCAR_mp-1001011'
    # pwd = r'./bulk_database/POSCAR_mp-1004037'
    # pwd = r'./bulk_database/POSCAR_mp-1056059'
    # pwd = r'./bulk_database/POSCAR_mp-1181286'
    pos = PosCar(pwd)
    cart = pos.dir2cart(pos.bases, pos.data)
    print('bases:', pos.bases)
    print('dir:', pos.data)
    print('cart:', cart)
    sr = SearchTetrahedron(pwd)
    print('abc:', sr.cal_lattice_lengths(pos.bases))
    print('ABC:', sr.cal_lattice_angles(pos.bases))

    # dis_array, coor_array = sr.get_all_distance_and_coord()
    # print(coor_array)

    # import matplotlib.pyplot as plt
    # for i in range(len(dis_array)):
    #     xx = []
    #     yy = []
    #     cc = dis_array[i].reshape(1, -1)
    #     cc = np.sort(cc)[0, 1:]
    #     print('cc:', cc)
    #     for idx, j in enumerate(cc):
    #         n = cc[0: idx + 1]
    #         std_n = np.std(n)
    #         n1 = cc[0: idx + 2]
    #         std_n1 = np.std(n1)
    #         # xx.append(std_n)
    #         # yy.append(std_n1)
    #         yy.append(std_n1-std_n)
    #     plt.scatter(list(range(len(yy))), yy)
    #     plt.show()
    # sr.cal_bonding_criterion(dis_array)
    genes_and_points = sr.search_rule()
    print(genes_and_points)
    # for i in genes_and_points:
    #     if i[0][0] == 3.482:
    #         print(i)
    print(len(genes_and_points))
    # print(genes_and_points)




