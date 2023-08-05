# -*- coding: utf-8 -*-

import os
import re
import time
import numba as nb
import numpy as np
import pandas as pd
from sympy import symbols, solve


@nb.jit(nopython=True)
def _dist(site1, site2):
    return np.linalg.norm(site1 - site2)


@nb.jit
def _angle(site1, site2, site3):
    m1 = site1 - site2
    m2 = site3 - site2
    m = m1.dot(m2) / (np.sqrt(m1.dot(m1)) * np.sqrt(m2.dot(m2)))
    if m > 1:
        m = 1
    elif m < -1:
        m = -1
    arc = np.arccos(m)
    return arc * 180 / np.pi


@nb.jit
def _di_angle(site1, site2, site3, site4):
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


@nb.jit(nopython=True)
def _gcd(m, n):
    m = abs(m)
    n = abs(n)
    m, n = (m, n) if m >= n else (n, m)
    while n:
        m, n = n, m % n
    return m


@nb.jit(nopython=True)
def _lcm(m, n):
    return m * n / _gcd(m, n)


@nb.jit
def _get_coor_from_dist(s1, s2, dist):
    s_12 = s2 - s1
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    # The x of the vector s_12 is 0, which will cause 0 to be divided by the x value of the vector s_12
    if (s_12[0] == 0) and (s_12[1] != 0) and (s_12[2] != 0):
        # solve x
        s_1w_x = solve(x - s1[0], x)[0]
        # solve y, z
        yz_list = solve([(s_1w_x - s1[0]) ** 2 + (y - s1[1]) ** 2 + (z - s1[2]) ** 2 - dist ** 2,
                         s_12[1] * (z - s1[2]) - s_12[2] * (y - s1[1])], [y, z])
        for yz_t in yz_list:
            # yz_t is a tuple
            if np.dot(np.array(yz_t) - s1[1:], s_12[1:]) < 0:
                continue
            else:
                # vector 12 and vector 1w have same direction
                s_1w_y = yz_t[0]
                s_1w_z = yz_t[1]
                w = np.array([s_1w_x, s_1w_y, s_1w_z]).astype('float64')
                return w
    # The y of the vector s_12 is 0
    elif (s_12[0] != 0) and (s_12[1] == 0) and (s_12[2] != 0):
        s_1w_y = solve(y - s1[1], y)[0]
        xz_list = solve([(x - s1[0]) ** 2 + (s_1w_y - s1[1]) ** 2 + (z - s1[2]) ** 2 - dist ** 2,
                         s_12[0] * (z - s1[2]) - s_12[2] * (x - s1[0])], [x, z])
        for xz_t in xz_list:
            if np.dot(np.array(xz_t) - np.array([s1[0], s1[2]]), np.array([s_12[0], s_12[2]])) < 0:
                continue
            else:
                s_1w_x = xz_t[0]
                s_1w_z = xz_t[1]
                w = np.array([s_1w_x, s_1w_y, s_1w_z]).astype('float64')
                return w
    # The z of the vector s_12 is 0
    elif (s_12[0] != 0) and (s_12[1] != 0) and (s_12[2] == 0):
        s_1w_z = solve(z - s1[2], z)[0]
        xy_list = solve([(x - s1[0]) ** 2 + (y - s1[1]) ** 2 + (s_1w_z - s1[2]) ** 2 - dist ** 2,
                         s_12[1] * (x - s1[0]) - s_12[0] * (y - s1[1])], [x, y])
        for xy_t in xy_list:
            if np.dot(np.array(xy_t) - s1[0:-1], s_12[0:-1]) < 0:
                continue
            else:
                s_1w_x = xy_t[0]
                s_1w_y = xy_t[1]
                w = np.array([s_1w_x, s_1w_y, s_1w_z]).astype('float64')
                return w
    # The x and y of the vector s_12 are 0
    elif (s_12[0] == 0) and (s_12[1] == 0) and (s_12[2] != 0):
        s_1w_x = solve(x - s1[0], x)[0]
        s_1w_y = solve(y - s1[1], y)[0]
        z_list = solve((s_1w_x - s1[0]) ** 2 + (s_1w_y - s1[1]) ** 2 + (z - s1[2]) ** 2 - dist ** 2, z)
        for z_t in z_list:
            if np.dot(np.array(z_t) - s1[2], s_12[2]) < 0:
                continue
            else:
                w = np.array([s_1w_x, s_1w_y, z_t]).astype('float64')
                return w
    # The x and z of the vector s_12 are 0
    elif (s_12[0] == 0) and (s_12[1] != 0) and (s_12[2] == 0):
        s_1w_x = solve(x - s1[0], x)[0]
        s_1w_z = solve(z - s1[2], z)[0]
        y_list = solve((s_1w_x - s1[0]) ** 2 + (y - s1[1]) ** 2 + (s_1w_z - s1[2]) ** 2 - dist ** 2, y)
        for y_t in y_list:
            if np.dot(np.array(y_t) - s1[1], s_12[1]) < 0:
                continue
            else:
                w = np.array([s_1w_x, y_t, s_1w_z]).astype('float64')
                return w
    # The y and z of the vector s_12 are 0
    elif (s_12[0] != 0) and (s_12[1] == 0) and (s_12[2] == 0):
        s_1w_y = solve(y - s1[1], y)[0]
        s_1w_z = solve(z - s1[2], z)[0]
        x_list = solve((x - s1[0]) ** 2 + (s_1w_y - s1[1]) ** 2 + (s_1w_z - s1[2]) ** 2 - dist ** 2, x)
        for x_t in x_list:
            if np.dot(np.array(x_t) - s1[0], s_12[0]) < 0:
                continue
            else:
                w = np.array([x_t, s_1w_y, s_1w_z]).astype('float64')
                return w
    # no zero in the vector s_12
    else:
        # Two parallel equations and one distance equation
        xyz_list = solve([s_12[1] * (x - s1[0]) - s_12[0] * (y - s1[1]),
                          s_12[2] * (x - s1[0]) - s_12[0] * (z - s1[2]),
                          (x - s1[0]) ** 2 + (y - s1[1]) ** 2 + (z - s1[2]) ** 2 - dist ** 2], [x, y, z])
        for xyz_t in xyz_list:
            if np.dot(np.array(xyz_t) - s1, s_12) < 0:
                continue
            else:
                s_1w_x = xyz_t[0]
                s_1w_y = xyz_t[1]
                s_1w_z = xyz_t[2]
                w = np.array([s_1w_x, s_1w_y, s_1w_z]).astype('float64')
                return w


def site_array(point):
    """
    convert coord str to numpy
    """
    coorstr = point.strip('[]')
    coorstr = coorstr.strip()
    alist = re.split('\s+', coorstr)
    return np.array([float(alist[0]), float(alist[1]), float(alist[2])])


class SearchMillerSurface:
    """
    get the miller index
    including calculating Greatest common divisor (gcd) and Least common multiple (lcm)
    """
    def __init__(self):
        """
        coord: 4×3 size,
        """

    def dist(self, site1, site2):
        """
        return the distance between site1 and site2
        """
        w = _dist(site1, site2)
        return w

    def angle(self, site1, site2, site3):
        """
        return the angle between of three coordinations
        """
        w = _angle(site1, site2, site3)
        return w

    def di_angle(self, site1, site2, site3, site4):
        """
        return the diangle between plane formed by site1, site2, site4 and plane formed by site1, site2, site3
        """
        w = _di_angle(site1, site2, site3, site4)
        return w

    def gcd(self, m, n):
        """
        calculating greatest common divisor, for example, the cd of 12 is: 1,2,3,4,6,12;
        the cd of 16 is: 1,2,4,8,16;
        thus gcd is 4
        16 % 12 -> 4, 12 % 4 -> 0, result = 4
        """
        w = _gcd(m, n)
        return w

    def lcm(self, m, n):
        """
        calculating least common multiple, for example, the cm of 12 is: 12, 24, 36, 48;
        the cm of 16 is: 16, 32, 48;
        thus lcm is 48
        lcm also equal to the multiple of two numbers, then divide gcd
        """
        w = _lcm(m, n)
        return w

    def get_miller_index(self, p1, p2, p3):
        """
        coord: 3×3 size, return the miller index of the surface formed by three points, which conform to the
        right-hand rule
        return the plane normal vector and hkl
        """
        a = p1
        b = p2
        c = p3
        ab = b - a
        ac = c - a
        # Plane normal vector
        n1 = np.cross(ab, ac)
        # Notably, 0.5, 2.5 will be reduced towards the nearest even number to 0 and 2
        n = n1 * 100
        n = np.round(n)
        n = n.astype('int')
        # conclude whether there is zero in plane normal vector
        if 0 in n:
            # s is the non-repeated normal vector value, bb is the number of each element in s
            s, bb = np.unique(n, return_counts=True)
            for zc, i in enumerate(s):
                if i == 0 and bb[zc] == 2:
                    # hkl contains two zero
                    hkl = n / n[np.nonzero(n)]
                    break
                if i == 0 and bb[zc] == 1:
                    # if just one zero, pass point and reduce are needed
                    no_zero = []  # get the index of nonzero values
                    [no_zero.append(w) for w, i in enumerate(n) if (i != 0)]
                    # the nonzero values of plane normal vector
                    w1 = n[no_zero[0]]
                    w2 = n[no_zero[1]]
                    # calculating the gcd of w1, w2
                    gg = self.gcd(w1, w2)
                    # reducing, for example, [2,2,0] -> [1,1,0]
                    hkl = np.array(n / gg)
                    if abs(w1) > 10 and abs(w2) > 10:
                        hkl = np.array([np.int(np.round(w1 / w2)), 1, 0])
                    break
                if i == 0 and bb[zc] == 3:
                    return n1, n
        else:
            # get the values of plane normal vector
            ff1 = n[0]
            ff2 = n[1]
            ff3 = n[2]
            # calculating gcd
            gcd_num = self.gcd(self.gcd(ff1, ff2), ff3)
            hkl = np.array(n / gcd_num)
        return n1, hkl.astype('int')

    def get_coor_from_dist(self, s1, s2, dist):
        """
        According to the coordinates of s1 and s2, determine the coordinates of point w from s1 to the length of dist,
        and s1 is the starting point
        return the coord of w
        """
        w = _get_coor_from_dist(s1, s2, dist)
        return w

    def get_miller_from_tetrahedron(self, site1, site2, site3, site4):
        """
        Enter the four points of the tetrahedron, and the distribution in space is: 4-1-2-3, where 1-2 is the
        surface atom, and 4-3 is the fragment molecule. Obtain the coordinates a, b and c of the three points forming
        the Miller surface, realized by vector ab, ac, and calculate Miller index.
        two situation:
            1.When forming a tetrahedron, positive da situation (site3 faces outwards):
             choose a at line 1-4, choose b at line 1-3, choose c at line 2-3;
             or choose a at line 2-3, choose b at line 2-4, choose c at line 1-4
             negative da situation:
             choose a at line 1-4, choose b at line 2-3, choose c at line 1-3;
             or choose a at line 2-3, choose b at line 1-4, choose c at line 2-4

            2.When forming a quadrilateral, only the Miller surface perpendicular to the plane (1,2,3,4) is considered.
              "a" is chosen at the 1-4 line segment, and traverses a point in the 2-3 segment as "b1",
              and site4 as "c1", First, find the normal vectors of a, b1, and c1; process the normal vectors,
              and take the value corresponding to "a" for all elements containing 0 to obtain "b";
              use point "b1" as "c" and calculate the normal vectors of a, b, and c, which is Miller Index
        """
        da = self.di_angle(site1, site2, site3, site4)
        ml_list = []  # miller index
        # quadrilateral
        if abs(da - 0) <= 10E-2:
            print("====Quadrilateral====")
            # get a
            dis = self.dist(site1, site2)
            angle1 = self.angle(site4, site1, site2)

            if angle1 < 90:
                adis = np.round(dis * np.sin((90-angle1)/180*np.pi), 3)
                try:
                    a = self.get_coor_from_dist(site1, site4, adis)
                    a = a.round(3)
                    b1 = site2.round(3)
                    site4 = site4.round(3)
                    for ai in [a, site1.round(3)]:
                        ab1 = b1 - ai
                        ac1 = site4 - ai
                        ml = np.cross(ab1, ac1)
                        b = np.zeros(3)
                        for idx, nx in enumerate(ml):
                            if nx == 0:
                                b[idx] = a[idx]
                            else:
                                b[idx] = nx
                        # n1 is the plane normal vector, ml is the relatively prime Miller index
                        n1, ml = self.get_miller_index(ai, b, b1)
                        ml = list(ml)
                        # get the low miller index
                        if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                            nc = np.unique(ml)
                            # ignore the [0,0,0] index
                            if (len(nc) == 1) and (nc[0] == 0):
                                pass
                            else:
                                ml_list.append(ml)
                                # print(len(ml_list))
                                # break
                except:
                    pass
            elif angle1 > 90:
                b1dis = np.round(dis * np.sin((angle1-90)/180*np.pi), 3)
                try:
                    b1 = self.get_coor_from_dist(site2, site3, b1dis)
                    b1 = b1.round(3)
                    a = site1.round(3)
                    site4 = site4.round(3)
                    for bi in [b1, site2.round(3)]:
                        ab1 = bi - a
                        ac1 = site4 - a
                        ml = np.cross(ab1, ac1)
                        b = np.zeros(3)
                        for idx, nx in enumerate(ml):
                            if nx == 0:
                                b[idx] = a[idx]
                            else:
                                b[idx] = nx
                        # n1 is the plane normal vector, ml is the relatively prime Miller index
                        n1, ml = self.get_miller_index(a, b, bi)
                        ml = list(ml)
                        # get the low miller index
                        if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                            nc = np.unique(ml)
                            # ignore the [0,0,0] index
                            if (len(nc) == 1) and (nc[0] == 0):
                                pass
                            else:
                                ml_list.append(ml)
                                # print(len(ml_list))
                                # break
                except:
                    pass
            else:
                a = site1.round(3)
                b1 = site2.round(3)
                site4 = site4.round(3)
                ab1 = b1 - a
                ac1 = site4 - a
                ml = np.cross(ab1, ac1)
                b = np.zeros(3)
                for idx, nx in enumerate(ml):
                    if nx == 0:
                        b[idx] = a[idx]
                    else:
                        b[idx] = nx
                # n1 is the plane normal vector, ml is the relatively prime Miller index
                n1, ml = self.get_miller_index(a, b, b1)
                ml = list(ml)
                # get the low miller index
                if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                    nc = np.unique(ml)
                    # ignore the [0,0,0] index
                    if (len(nc) == 1) and (nc[0] == 0):
                        pass
                    else:
                        ml_list.append(ml)
                        # print(len(ml_list))
                        # break

        # tetrahedron, da is positive, line2-3 out of surface
        elif da > 0:
            print("====tetrahedron，positive da====")
            # get a, b, c
            dis14 = self.dist(site1, site4)
            dis13 = self.dist(site1, site3)
            dis23 = self.dist(site2, site3)
            dis24 = self.dist(site2, site4)
            f_14_23 = self.get_coor_from_dist(site1, site4, 2 / 3 * dis14)
            f_14_12 = self.get_coor_from_dist(site1, site4, 1 / 2 * dis14)
            f_14_13 = self.get_coor_from_dist(site1, site4, 1 / 3 * dis14)
            f_23_23 = self.get_coor_from_dist(site2, site3, 2 / 3 * dis23)
            f_23_12 = self.get_coor_from_dist(site2, site3, 1 / 2 * dis23)
            f_23_13 = self.get_coor_from_dist(site2, site3, 1 / 3 * dis23)
            f_13_23 = self.get_coor_from_dist(site1, site3, 2 / 3 * dis13)
            f_13_12 = self.get_coor_from_dist(site1, site3, 1 / 2 * dis13)
            f_13_13 = self.get_coor_from_dist(site1, site3, 1 / 3 * dis13)
            f_24_23 = self.get_coor_from_dist(site2, site4, 2 / 3 * dis24)
            f_24_12 = self.get_coor_from_dist(site2, site4, 1 / 2 * dis24)
            f_24_13 = self.get_coor_from_dist(site2, site4, 1 / 3 * dis24)
            a_list = [site4, f_14_23, f_14_12, f_14_13]
            b_list = [f_13_23, f_13_12, f_13_13, site1]
            c_list = [f_23_23, f_23_12, f_23_13, site2]

            for a in a_list:
                for b in b_list:
                    for c in c_list:
                        a = np.round(a, 3)
                        b = np.round(b, 3)
                        c = np.round(c, 3)
                        # n1 is the plane normal vector, ml is the relatively prime Miller index
                        n1, ml = self.get_miller_index(a, b, c)
                        ml = list(ml)
                        if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                            nc = np.unique(ml)
                            # ignore the [0,0,0] index
                            if (len(nc) == 1) and (nc[0] == 0):
                                pass
                            else:
                                ml_list.append(ml)
            # continue
            a_list = [site3, f_23_23, f_23_12, f_23_13]
            b_list = [f_24_23, f_24_12, f_24_13, site2]
            c_list = [f_14_23, f_14_12, f_14_13, site1]

            for a in a_list:
                for b in b_list:
                    for c in c_list:
                        a = np.round(a, 3)
                        b = np.round(b, 3)
                        c = np.round(c, 3)
                        # n1 is the plane normal vector, ml is the relatively prime Miller index
                        n1, ml = self.get_miller_index(a, b, c)
                        ml = list(ml)
                        if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                            nc = np.unique(ml)
                            # ignore the [0,0,0] index
                            if (len(nc) == 1) and (nc[0] == 0):
                                pass
                            else:
                                ml_list.append(ml)

        # tetrahedron, da is negative, line2-4 out of surface
        else:
            print("====tetrahedron，negative da====")
            # get a, b, c
            dis14 = self.dist(site1, site4)
            dis13 = self.dist(site1, site3)
            dis23 = self.dist(site2, site3)
            dis24 = self.dist(site2, site4)
            f_14_23 = self.get_coor_from_dist(site1, site4, 2 / 3 * dis14)
            f_14_12 = self.get_coor_from_dist(site1, site4, 1 / 2 * dis14)
            f_14_13 = self.get_coor_from_dist(site1, site4, 1 / 3 * dis14)
            f_23_23 = self.get_coor_from_dist(site2, site3, 2 / 3 * dis23)
            f_23_12 = self.get_coor_from_dist(site2, site3, 1 / 2 * dis23)
            f_23_13 = self.get_coor_from_dist(site2, site3, 1 / 3 * dis23)
            f_13_23 = self.get_coor_from_dist(site1, site3, 2 / 3 * dis13)
            f_13_12 = self.get_coor_from_dist(site1, site3, 1 / 2 * dis13)
            f_13_13 = self.get_coor_from_dist(site1, site3, 1 / 3 * dis13)
            f_24_23 = self.get_coor_from_dist(site2, site4, 2 / 3 * dis24)
            f_24_12 = self.get_coor_from_dist(site2, site4, 1 / 2 * dis24)
            f_24_13 = self.get_coor_from_dist(site2, site4, 1 / 3 * dis24)
            a_list = [site4, f_14_23, f_14_12, f_14_13]
            b_list = [f_23_23, f_23_12, f_23_13, site2]
            c_list = [f_13_23, f_13_12, f_13_13, site1]

            for a in a_list:
                for b in b_list:
                    for c in c_list:
                        a = np.round(a, 3)
                        b = np.round(b, 3)
                        c = np.round(c, 3)
                        # n1 is the plane normal vector, ml is the relatively prime Miller index
                        n1, ml = self.get_miller_index(a, b, c)
                        ml = list(ml)
                        if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                            nc = np.unique(ml)
                            # ignore the [0,0,0] index
                            if (len(nc) == 1) and (nc[0] == 0):
                                pass
                            else:
                                ml_list.append(ml)
            # continue
            a_list = [site3, f_23_23, f_23_12, f_23_13]
            b_list = [f_14_23, f_14_12, f_14_13, site1]
            c_list = [f_24_23, f_24_12, f_24_13, site2]

            for a in a_list:
                for b in b_list:
                    for c in c_list:
                        a = np.round(a, 3)
                        b = np.round(b, 3)
                        c = np.round(c, 3)
                        # n1 is the plane normal vector, ml is the relatively prime Miller index
                        n1, ml = self.get_miller_index(a, b, c)
                        ml = list(ml)
                        if np.max(abs(np.array(ml))) - 3 < 10E-1 and (ml not in ml_list):
                            nc = np.unique(ml)
                            # ignore the [0,0,0] index
                            if (len(nc) == 1) and (nc[0] == 0):
                                pass
                            else:
                                ml_list.append(ml)

        return ml_list

    def main(self, filename, outfile='final_exist_index.csv'):
        df = pd.read_csv(filename)
        sms_list = []
        have_index = 0
        point1 = df['point1'].values
        point2 = df['point2'].values
        point3 = df['point3'].values
        point4 = df['point4'].values
        for n in range(len(df)):
            print('Screening {}'.format(n))
            p1 = site_array(point1[n])
            p2 = site_array(point2[n])
            p3 = site_array(point3[n])
            p4 = site_array(point4[n])
            sms = self.get_miller_from_tetrahedron(p1, p2, p3, p4)
            print(sms)
            if len(sms):
                sms_list.append(sms)
                have_index += 1
            else:
                sms_list.append('None')
        print('total size:', len(df))
        print('index size:', len(sms_list))
        print('have index numbers:', have_index)
        df['indx'] = sms_list
        df.to_csv(outfile, index=False)
        print('Done.')


if __name__ == '__main__':
    # pwd = os.getcwd()
    # nn = os.path.basename(pwd)
    # SearchMillerSurface().main(filename=r'final_exist_{}.csv'.format(nn), outfile=r'final_exist_index_{}.csv'.format(nn))

    start = time.time()
    print(start)
    df = pd.read_csv('all_data.csv')
    # df = pd.read_csv('final_exist.csv')
    df = df[df['id'] == 'mp-1336']
    # print(df.values)
    point1 = df['point1'].values
    point2 = df['point2'].values
    point3 = df['point3'].values
    point4 = df['point4'].values
    for i in range(len(df)):
        # if i == 2:
            a = site_array(point1[i])
            b = site_array(point2[i])
            c = site_array(point3[i])
            d = site_array(point4[i])
            print(a)
            print(b)
            print(c)
            print(d)
            sms = SearchMillerSurface().get_miller_from_tetrahedron(a, b, c, d)
            print(sms)
    end = time.time()
    print(end)
    result = end-start
    print(result)