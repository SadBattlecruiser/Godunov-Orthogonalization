import numpy as np
import pandas as pd
from scipy.integrate import ode


# Численный интегратор
def nintegrate(F, y0, t0, tk, steps):
    intgr = ode(F).set_integrator('Dopri5') # Рунге-Кутта 4-5
    intgr.set_initial_value(y0, t0)
    time_range = np.linspace(t0, tk, steps)
    dt = (tk - t0)/(steps - 1)
    # Интегрируем. Он каждый шажок делает отдельно
    result_temp = np.array([np.append(np.array([t0]), y0)])
    for i in range(steps):
        t_curr = intgr.t
        y_curr = intgr.integrate(intgr.t+dt)
        row_curr = [np.append(np.array([t_curr]), y_curr)]
        result_temp = np.append(result_temp, row_curr, axis=0)
    result_np = np.array(result_temp)
    return result_np

# Функция, делающая по векторам ГУ слева и справа матрицу ГУ слева для метода начальных параметров
def Y0_create(y_begin_, y_end_):
    y_begin = np.array(y_begin_)
    y_end = np.array(y_end_)
    dim = len(y_begin)
    # Индексы тех элементов, которые мы задали как известные
    begin_ind = np.where(~np.isnan(y_begin))[0]
    end_ind = np.where(~np.isnan(y_end))[0]
    # И тех, которые задали неизвестными
    begin_nan_ind = np.where(np.isnan(y_begin))[0]
    end_nan_ind = np.where(np.isnan(y_end))[0]
    # Количество заданных ГУ слева и справа
    begin_num = begin_ind.size
    end_num = end_ind.size
    # Создаем матрицу
    Y0 = np.zeros((dim, end_num + 1))
    Y0[:, end_num][begin_ind] = y_begin[begin_ind] # Это для частного неоднородного, y0
    for i in range(end_num):
        Y0[:, i][begin_nan_ind[i]] = 1
    return Y0

# Функция, которая принимает Y-матрицу в начале участка и возвращает в конце
# Y-матрица в numpy-виде
def Y_nintegrate(F_, F_plus_g_, Y0_, t0_, tk_, steps_):
    dim_n = Y0_.shape[0]     # Размерность вектора состояния, т.е. высота матрицы
    dim_r = Y0_.shape[1] - 1 # Количество решений однородного, т.е. ширина матрицы - 1
    Yk = np.copy(Y0_)
    Yk_full = []
    for i in range(dim_r):
        Yk_full.append(nintegrate(F_, Y0_[:, i], t0_, tk_, steps_))
        Yk[:, i] = Yk_full[i][steps_, 1:]
    Yk_full.append(nintegrate(F_plus_g_, Y0_[:, dim_r], t0_, tk_, steps_))
    Yk[:, dim_r] = Yk_full[dim_r][steps_, 1:]
    return [Yk, Yk_full]

def set_Wij(Y_, W_, Z_, i_, j_):
    r = Y_.shape[1]
    i = i_
    j = j_
    if (i == j):
        if (i == r - 1):
            W_[i, j] = 1
        else:
            W_[i, j] = np.sqrt(Y_[:, i].dot(Y_[:, i]) - np.sum(np.power(W_[:, i], 2)[range(i)]))
    else:
        W_[i, j] = Z_[:, i].dot(Y_[:, j])
    #print('set_Wij, i, j:', i, j, W_[i, j])

def set_Zi(Y_, W_, Z_, i_):
    r = Y_.shape[1]
    i = i_
    ZW_sum = 0.
    for j in range(i):
        ZW_sum += Z_[:, j]*W_[j, i]
    if (i == r - 1):
        Z_[:, i] = (Y_[:, i] - ZW_sum)
    else:
        Z_[:, i] = 1/W_[i, i] * (Y_[:, i] - ZW_sum)
    #print('set_Zi, i:', i)

# Функия, формирующая саму новую ортогонализованную матрицу
# И омега-матрицу для оротогонализации
def orthogonalization(Y_):
    dim_n = Y_.shape[0]     # Размерность вектора состояния, т.е. высота матрицы
    dim_r = Y_.shape[1]     # Количество векторов однородного, т.е. ширина матрицы - 1
    W = np.zeros((dim_r, dim_r))
    Z = np.zeros((dim_n, dim_r))
    for i in range(dim_r):
        for j in range(i+1):
            set_Wij(Y_, W, Z, j, i)
        set_Zi(Y_, W, Z, i)
    return [Z, W]

# Функция, которая решает СЛАУ и находит вектор констант
# C_vec = [C1, C2, ..., Cr, 1]
def solve_c0_vec(Y_, y_end_):
    # Индексы тех элементов, которые мы задали как известные
    end_ind = np.where(~np.isnan(y_end_))[0]
    # И тех, которые задали неизвестными
    end_nan_ind = np.where(np.isnan(y_end_))[0]
    # Количество заданных ГУ справа
    end_num = end_ind.size
    # Теперь составляем слау и находим константы
    # y0k + Y.dot(c_vec) = y_end -> Y.dot(c_vec) = (y_end - y0k)
    Y_matr = Y_[:, :-1][end_ind]
    y0k = Y_[:, -1]
    r_vec = y_end_[end_ind] - y0k[end_ind]
    c_vec = np.linalg.solve(Y_matr, r_vec)
    c_vec_full = np.append(c_vec, 1.)
    return c_vec_full

def solve_ci_vec(Wi, c_prev):
    dim_r = c_prev.shape[0] - 1
    c_i = np.zeros(dim_r + 1)
    for i in range(dim_r + 1):
        c_sum = 0
        for j in range(i):
            c_sum += Wi[-1-i, -1-(i-j-1)]*c_i[-1-(i-j-1)]
        c_i[-1-i] = 1/Wi[-1-i, -1-i] * (c_prev[-1-i] - c_sum)
    return c_i

# Функция, собирающая нормальную функцию y
# из Y_full_list и c_list
# Нужна для Годунова, см. следующую функцию
# Возвращает красивый pandas.DataFrame
def compile_y_func(Y_full_list, c_list_):
    c_list = c_list_.copy()
    c_list.reverse() # Он был задом наперед
    num_of_parts = len(c_list)
    num_of_c = c_list[0].shape[0] # Сколько констант в одном векторе из c_list
    num_of_steps = Y_full_list[0][0].shape[0] # Сколько шагов интегрирования в каждом part
    num_of_func = Y_full_list[0][0].shape[1]-1 # Размерноесть вектора состояния, -1 ибо один столбец t
    # Имена для столбцов функций в DataFrame
    names = ['t']
    for i in range(num_of_func):
        names.append('y' + str(i))
    # Лист с DataFrame каждого part
    y_df_list = []
    for i in range(num_of_parts):
        time_vec_curr = Y_full_list[i][0][:,0]
        y_data_curr = np.zeros((num_of_steps, num_of_func))
        for j in range(num_of_c):
            y_data_curr += Y_full_list[i][j][:,1:] * c_list[i][j]
        y_data_curr_with_t = np.append(time_vec_curr.reshape((time_vec_curr.size, 1)), y_data_curr, axis=1)
        y_df_curr = pd.DataFrame(y_data_curr_with_t, columns=names)
        y_df_list.append(y_df_curr)
    # И лепим все фреймы в первый
    for i in range(1, len(y_df_list)):
        y_df_list[0] = y_df_list[0].append(y_df_list[i])
    return y_df_list[0].drop_duplicates('t')

# Собственно ортогонализация и есть
# Возвращает DataFrame с приблизительным решением
def godunov_orthogonalization_solve(F_, F_plus_g_, y_begin_, y_end_, t0_, tk_, parts_, steps_):
    y_begin = np.array(y_begin_)
    y_end = np.array(y_end_)
    t_range = np.linspace(t0_, tk_, num=parts_+1)
    #print('t_range:', t_range)
    Y0 = Y0_create(y_begin, y_end)
    # Листы матриц решений и матриц ортогонализации
    Y_list = [Y0]
    W_list = []
    Y_full_list = []
    for i in range(parts_):
        print('Integrate part', i)
        Yi_temp, Yi_full_temp = Y_nintegrate(F_, F_plus_g_, Y_list[i], t_range[i], t_range[i+1], steps_)
        if (i != parts_ - 1):
            Yi, Wi = np.linalg.qr(Yi_temp)
            Yi[:, -1] = Yi[:, -1] * Wi[-1, -1]
            Wi[-1, -1] = 1.

            #Yi, Wi = orthogonalization(Yi_temp)
            Y_list.append(Yi)
            Y_full_list.append(Yi_full_temp)
            W_list.append(Wi)
        else:
            Y_list.append(Yi_temp)
            Y_full_list.append(Yi_full_temp)
    # Получаем лист векторов состояния по краям parts начиная с последнего
    print('Сalculate c_list')
    c_list = [solve_c0_vec(Y_list[-1], y_end)]
    for i in range(parts_ - 1):
        c_list.append(solve_ci_vec(W_list[-(1+i)], c_list[i]))
    # Теперь формируем по данным DataFrame, который и вернем
    print('Compile y_func')
    ret_res = compile_y_func(Y_full_list, c_list)
    return ret_res
