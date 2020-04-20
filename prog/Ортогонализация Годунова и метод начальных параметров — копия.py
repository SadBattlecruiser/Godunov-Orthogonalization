import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from godunov_orthogonalization import godunov_orthogonalization_solve


###########
def r(t):
    return r1 + t*np.cos(th)

def Fmatr(t):
    th = pi/2 - np.arctan((r1-r1) / L)
    print('th', th)
    ###
    ret_matr = np.matrix([
    [-mu*np.cos(th)/r(t), -mu*k/r(t), -1/r1 - mu*np.sin(th)/r(t), 0, (1 - mu**2)/(E*h*r(t)), 0, 0, 0],                          #1
    [k/r(t), np.cos(th)/r(t), 0, 0, 0, 2*(1 + mu)/(E*h*r(t)), 0, 0],                                                            #2
    [1/r1, 0, 0, -1, 0, 0, 0, 0],                                                                                               #3
    [0, -mu*k/(r(t)**2)*np.sin(th), -mu*(k**2)/(r(t)**2), -mu*np.cos(th)/r(t), 0, 0, 0, 12*(1 - mu**2)/(E*(h**3)*r(t))],        #4
    [E*h/r(t) * ((np.cos(th))**2 + (k*h*np.sin(th))**2 / (6*(1 + mu)*(r(t)**2))), k*E*h/r(t) * np.cos(th),                      #5
        E*h/r(t) * np.sin(th)*np.cos(th) * (1 - (k*h)**2 / (6*(1 + mu)*(r(t)**2))),
        -(k**2)*E*(h**3) / (6*(1 + mu)*(r(t)**2)) * np.sin(th), mu/r(t) * np.cos(th), -k/r(t), -1/r1, 0,],
    [E*h/r(t) * k*np.cos(th), (k**2)*E*h/r(t), E*h/r(t) * k*np.sin(th) * (1 + ((k*h)**2) / (12 * r(t)**2)),                     #6
        E*(h**3) / (12*(r(t)**3)) * k*np.sin(th)*np.cos(th), mu*k/r(t), -np.cos(th)/r(t), 0, mu*k / (r(t)**2) * np.sin(th)],
    [E*h/r(t) * np.sin(th)*np.cos(th) * (1 - (k*h)**2 / (6*(1 + mu)*(r(t)**2))),                                                #7
        E*h/r(t) * k*np.sin(th) * (1 + ((k*h)**2) / (12*(r(t)**2))), E*h/r(t) * (np.sin(th)**2 + (k**4)*(h**2)/(12*(r(t)**2))),
        (3+mu)/(1+mu) * E*(h**3)/(12*(r(t)**2)) * (k**2) * np.cos(th), 1/r1 + mu*np.sin(th)/r(t), 0, 0, mu*(k/r(t))**2],
    [-(k**2)*E*(h**3) / (6*(1 + mu)*(r(t)**2)) * np.sin(th), E*(h**3) / (12*(r(t)**2)) * k*np.sin(th)*np.cos(th),               #8
        (3+mu)/(1+mu) * E*(h**3)/(12*(r(t)**2)) * (k**2) * np.cos(th),
        E*(h**3) / (12*r(t)) * (np.cos(th)**2 + 2*(k**2)/(1+mu)), 0, 0, 1, mu/r(t) * np.cos(th)],
    ])
    return ret_matr


def g_vec(t):
    return load_coeff * np.array([0., 0., 0., 0., 0., 0., -r(t), 0.])
###########



# Специальная штука, которую передаем интегратору
# Оставь как есть
def F(t, y):
    return (Fmatr(t).dot(y))
def F_plus_g(t, y):
    return (Fmatr(t).dot(y)) + g_vec(t)

###########
# Даешь данные задачи
pi = 3.14159265
h = 2.
L = 400.
r1 = 200.
r2 = 400.
ro = 1000 * 0.001**3
mu = 0.3
R1 = 1000000000000. # Это такая бесконечность
E = 2e5
g = 9.81
# Коэффициенты разложения в ряд нагрузки
# !!! Если 0 - поставь 0, не пропускай
harmonics_coeffs = [-g*ro*r/pi, 2, 3]
###########
# Параметры
s0 = 0                                              # Это начало интервала интегрирования
smax = L / np.sin(pi/2 - np.arctan((r2-r1) / L))    # Это конец интервала интегрирования
parts = 20                                          # Это на сколько участков делим
steps = 4000                                        # Это сколько шагов интегрирования на участок
# Начальные условия. Которые не знаем - пиши как float('nan')
y_begin = [0., 0., 0., 0., float('nan'), float('nan'), float('nan'), float('nan')]
y_end = [0., 0., 0., 0., float('nan'), float('nan'), float('nan'), float('nan')]
###########

harmonics_coeffs_np = np.array(harmonics_coeffs)
num_of_harmonics =len(harmonics_coeffs)
print('Количество гармоник:\n', num_of_harmonics)
print('Значения разложения нагрузок:\n', harmonics_coeffs_np)
print('Параметры установлены\n')
for i in range(num_of_harmonics):
    if (harmonics_coeffs_np[i] != 0.):
        k = i + 1
        load_coeff = harmonics_coeffs_np[i]
        print('Текущая гармоника:', k)
        yk_func_res = godunov_orthogonalization_solve(F, F_plus_g, y_begin, y_end, s0, smax, parts, steps)
        file_name = r'out\harmonic' + str(k) + '.csv'
        print(file_name)
        yk_func_res.to_csv(file_name, index = False)
