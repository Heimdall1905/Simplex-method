import numpy as np
from scipy.optimize import linprog

# на данный момент принимает данные для минимизации функции
def simplex(c, A, b, text = False):
    n, m = A.shape # размеры матрицы А
    m0 = m

    A = np.hstack((A, np.eye(n))) # добавляем переменные
    c = np.hstack((c, np.zeros(n)))

    # теперь нужно заполнить симплексную таблицу, если было n строк и m столбцов, то теперь будет n + 1 строк и m + n + 1 столбцов
    my_table = np.zeros((n + 1, n + m + 1))

    A = np.column_stack((A, b))
    c = np.append(c, 0)
    my_table = np.concatenate(([c], A))

    # нужно учесть базисы для нахождения точек
    left_basis = [f"s{i}" for i in range(n)]
    top_basis = [f"X{i}" for i in range(m)] + [f"s{i}" for i in range(n)]

    # обновляем размерность
    n, m = my_table.shape

    my_table, left_basis, top_basis = simplex_k(my_table, left_basis, top_basis, False, text)
    # возвращаем решение
    '''solu = {}
    for i in range(m0):
        if f'X{i}' in left_basis:
            solu[f'X{i}'] = my_table[i + 1, -1]
        else:
            solu[f'X{i}'] = 0'''

    return my_table[0, -1] * -1 # для минимума

def two_phase_simplex(c, A, b, bounds, tag = 'min', text = False):
    if tag == 'max':
        index = 1
    else:
        index = -1

    e = bounds.count('E')
    g = bounds.count('G')
    l = bounds.count('L')

    c = c * -index
    n, m = A.shape  # размеры матрицы А
    roots = e + 2*g + l # количество добавляемых столбцов
    art_roots = []
    A = np.hstack((A, np.zeros((n, roots))))  # добавляем переменные
    c = np.hstack((c, np.zeros(roots)))
    numbers = np.zeros(m + roots + 1)
    a = np.zeros(m + roots + 1)

    left_basis = []
    top_basis = [f"X{i}" for i in range(m)] + [f"{i}" for i in range(len(c) - m)]

    ptr = -1
    si = 1
    Ri = 1
    for i in range(n):
        if bounds[i] == 'L':
            A[i, m + i] = 1
            top_basis[m + i] = f"s{si}"
            left_basis.append(f"s{si}")
            si += 1

        if bounds[i] == 'G':
            A[i, m + i] = 1
            a[m + i] = 1
            a0 = np.zeros(n)
            a0[i] = -1
            A[:, ptr] = a0

            numbers += np.append(-A[i], -b[i])
            art_roots.append(m + i)
            top_basis[m + i] = f"R{Ri}"
            left_basis.append(f"R{Ri}")
            Ri += 1
            top_basis[ptr] = f"s{si}"
            si += 1
            ptr -= 1
        if bounds[i] == 'E':
            A[i, m + i] = 1
            a[m + i] = 1
            numbers += np.append(-A[i], -b[i])
            art_roots.append(m + i)
            top_basis[m + i] = f"R{Ri}"
            left_basis.append(f"R{Ri}")
            Ri += 1

    numbers += a
    my_table = np.zeros((n + 2, m + roots + 1))

    my_table[2:n + 2, :-1] = A
    my_table[2:n + 2, -1] = b
    my_table[1, :-1] = c
    my_table[0, :] = numbers

    my_table, left_basis, top_basis = simplex_k(my_table, left_basis, top_basis, True, text)

    if my_table[0, -1] < 0:
        print("В исходной задачи нет допустимого решения")
    if my_table[0, -1] == 0 and any('R' in i for i in left_basis):
        print("Система ограничений избыточна")

    # удаляем искусственные переменные и дополнительную функцию
    a_ = []
    for j in top_basis:
        if "R" in j:
            continue
        else:
            a_.append(j)
    top_basis = a_

    my_table = np.delete(my_table, 0, axis= 0)
    my_table = np.delete(my_table, art_roots, axis= 1)

    my_table, left_basis, top_basis = simplex_k(my_table, left_basis, top_basis, False, text)
    return my_table[0, -1] * index

def simplex_k(my_table, left_basis, top_basis, two = False, text = False):
    n, m = my_table.shape
    ptr = 1
    if two:
        index = 2
    else:
        index = 1
    while True:
        if text:
            print(f"Итерация номер {ptr}, текущая симпликсная таблица:\n", my_table)

        # условие оптимальности
        #my_table = np.round(my_table, 3)
        if np.all(my_table[0, :-1] >= 0):
            break

        # ищем ведущий столбец
        s = np.argmin(my_table[0, :-1])

        # ограниченность задачи
        if np.all(my_table[1:n, s] <= 0):
            print("Ошибка: значение задачи не ограничено.")
            break

        # ищем ведущую строку
        arr = []
        for i in range(index, n):
            if my_table[i, s] <= 0:
                arr.append(np.inf)
                continue
            arr.append(my_table[i, -1] / my_table[i, s])
        r = np.argmin(arr) + index

        my_table[r, :] /= my_table[r, s]  # привели к 1 ведущий элемент, теперь нужно по гауссить
        for i in range(n):
            if i == r:
                continue
            my_table[i, :] -= my_table[r, :] * my_table[i, s]

        # меняем базисные переменные
        h = left_basis[r - index]
        # print(r, s)
        left_basis[r - index] = top_basis[s]

        ptr += 1
    return my_table, left_basis, top_basis

'''c_ub = np.array([4, 1])

A_ub = np.array([[6, 2], [8, 6], [2, 4]])

b_ub = np.array([5, 7, 3])

bounds = ['E', 'G', 'L']

#solu, basis = simplex(c, A_ub, b_ub)
solu = two_phase_simplex(c_ub, A_ub, b_ub, bounds, tag= 'min')
result = linprog(c_ub, A_ub=A_ub, b_ub=b_ub, method='highs')
print(solu, result.fun)'''





