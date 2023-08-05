# -*- coding: utf-8 -*-

import os
import pandas as pd
from ase.io import vasp
from vaspy.atomco import PosCar
from ase.constraints import FixAtoms
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar


# the current path
pwd = os.getcwd()
# 42 type elements
TransitionMetal = ['Mg', 'Al', 'Si',
                   'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
                   'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Ba', 'La', 'Ce', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Bi']

data_list = []  # save all the material_id
name_list = []  # save all the chemical formula
spacegroup_list = []    # save all the spacegroup
# API key
with MPRester("KW3XBL80E7eOrU1i") as m:
    # data = m.query(criteria={"elements": {"$in": TransitionMetal, "$all": ["O"]}, "nelements": {"$lte": 3}},
    # less and equal to three elements
    data = m.query(criteria={"elements": {"$all": ["O"]}, "nelements": {"$lte": 3}},
                   properties=["unit_cell_formula", "pretty_formula", "material_id", "elements", "spacegroup", "nelements"])
    for i in data:
        finthon = True
        elements = i['elements']
        for j in elements:
            if j not in (TransitionMetal + ['O']):
                finthon = False
                break
        if finthon is False:
            continue
        else:
            data_list.append(i['material_id'])
            name_list.append((i['pretty_formula']))
            spacegroup_list.append(i['spacegroup'])

# save
df = pd.DataFrame()
df['pretty_formula'] = name_list
df['material_id'] = data_list
df['space_group'] = spacegroup_list
df.to_excel('test.xlsx', index=False)
# print the number
print(len(data_list))

# create bulk_database folder
database_pwd = os.path.join(pwd, 'bulk_database')
if not os.path.exists(database_pwd):
    os.makedirs(database_pwd)
# save POSCAR into bulk_database
for n, i in enumerate(data_list):
    if ('POSCAR_' + i) in os.listdir(database_pwd):
        continue
    struct = m.get_structure_by_material_id(i)
    # sort by electronegativity, from low to high
    struct.sort()
    # pymatgen save_function
    with open('POSCAR_temp', 'w') as f:
        f.write(str(Poscar(struct)))
    # ase save_function, and add constraint
    struct = vasp.read_vasp('POSCAR_temp')
    c = FixAtoms(indices=[n for n, i in enumerate(struct.get_positions()[:, 2])])
    struct.set_constraint(c)
    vasp.write_vasp('POSCAR_temp', struct, direct=True, vasp5=True)
    # vaspy save_function (just for beautiful format)
    pos_dir = os.path.join(database_pwd, 'POSCAR_{}'.format(i))
    PosCar('POSCAR_temp').tofile(pos_dir)
    print("{} done.".format(n+1))
    os.remove('POSCAR_temp')