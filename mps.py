from pulp import LpProblem
from scipy.optimize import linprog
import numpy as np

import simplex_method as sp

def parse_bounds(file_path):
    with open(file_path, 'r') as file:
        bounds = []
        current_section = None

        for line in file:
            line = line.strip()
            if line.startswith("BOUNDS"):
                current_section = "BOUNDS"
                continue
            elif line.startswith("END"):
                break

            if current_section == "BOUNDS":
                parts = line.split()
                bound_type = parts[0]  # LO, UP, FX
                variable_name = parts[2]
                value = float(parts[3])

                bounds.append((bound_type, variable_name, value))



        return bounds

# Задаем имя файла
mps_file = 'mik-250-20-75-4.mps'
tag = 'min'

# Чтение модели из MPS файла
problem = LpProblem.fromMPS(mps_file)

A_ub = []
A_ub_l = []
A_ub_l_e = []
b_ub = np.array([])
b_ub_l = np.array([])
b_ub_l_e = np.array([])
c_ub = np.array([])
main_bound = []

# заполнение с
for name in problem[1].variables():
    if problem[1].objective.get(name) is not None:
        c_ub = np.append(c_ub, problem[1].objective[name])
    else:
        c_ub = np.append(c_ub, [0])

a = []
R_names = []
for name, con in problem[1].constraints.items():
    b_ub = np.append(b_ub, [-con.constant]) # правая часть
    #print(name, con) # print номер строки/строка
    a = [] # коэффициенты строки
    R_names.append(name)
    if con.sense == -1:
        main_bound.append('L')
        b_ub_l = np.append(b_ub_l, [-con.constant])
        l_index = -1
    if con.sense == 0:
        main_bound.append('E')
        b_ub_l_e = np.append(b_ub_l_e, [-con.constant])
        l_index = 0
    if con.sense == 1:
        main_bound.append('G')
        b_ub_l = np.append(b_ub_l, [con.constant])
        l_index = 1
    for name in problem[1].variables():
        if con.get(name) is not None:
            a.append(con[name])
        else:
            a.append(0)

    A_ub.append(a)
    if l_index == -1:
        A_ub_l.append(a)
    if l_index == 0:
        A_ub_l_e.append(a)
    if l_index == 1:
        A_ub_l.append([-i for i in a])



m = len(a) # количество столбцов

C_i = [str(i) for i in problem[1].variables()] # все переменные
linprog_bounds = [[0, None] for _ in range(len(C_i))]

# ограничения
bounds = parse_bounds(mps_file)

for i in bounds:
    a = np.zeros(m)
    if i[0] == 'UP':
        a[C_i.index(i[1])] = 1
        linprog_bounds[C_i.index(i[1])][1] = i[2]
        b_ub = np.append(b_ub, [i[2]])
        A_ub.append(a)
        main_bound.append('L')
    if i[0] == 'LO':
        a[C_i.index(i[1])] = 1
        linprog_bounds[C_i.index(i[1])][0] = i[2]
        b_ub = np.append(b_ub, [i[2]])
        A_ub.append(a)
        main_bound.append('G')
linprog_bounds = [(i[0], i[1]) for i in linprog_bounds]

A_ub = np.array(A_ub)
l, l1 = A_ub.shape




if main_bound.count('L') == len(main_bound):
    ans = sp.simplex(c_ub, A_ub, b_ub)
else:
    ans = sp.two_phase_simplex(c_ub, A_ub, b_ub, main_bound, tag= 'min')

print(f"Наш ответ: {ans}")

if len(b_ub_l_e) == 0:
    result = linprog(c_ub, A_ub=A_ub_l, b_ub=b_ub_l, bounds=linprog_bounds, method='highs')
else:
    result = linprog(c_ub, A_ub=A_ub_l, b_ub=b_ub_l, A_eq= A_ub_l_e, b_eq= b_ub_l_e, bounds= linprog_bounds, method='highs')
print("Ответ из scipy.optimize: ", end='')
if result.success:
    print(result.fun)
else:
    print("решение не найдено, ", result.message)













