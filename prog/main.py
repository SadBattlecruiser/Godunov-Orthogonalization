import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from godunov_orthogonalization import godunov_orthogonalization_solve


###########
def r(t):
    #print('inside r(t) np.arctan((r2-r1) / L)', np.arctan((r2-r1) / L))
    return r1 + t*np.sin(np.arctan((r2-r1) / L))
###########
class Diff_eq:
    #"""docstring"""
    def __init__(self, k_):
        #"""Constructor"""
        self.k = k_
    def Fmatr(self, t):
        th = pi/2 - np.arctan((r2-r1) / L)
        ###
        ret_matr = np.matrix([
            [-mu*np.cos(th)/r(t), -mu*self.k/r(t), -1/R1 - mu*np.sin(th)/r(t), 0, (1 - mu**2)/(E*h*r(t)), 0, 0, 0],                                     #1
            [self.k/r(t), np.cos(th)/r(t), 0, 0, 0, 2*(1 + mu)/(E*h*r(t)), 0, 0],                                                                       #2
            [1/R1, 0, 0, -1, 0, 0, 0, 0],                                                                                                               #3
            [0, -mu*self.k/(r(t)**2)*np.sin(th), -mu*(self.k**2)/(r(t)**2), -mu*np.cos(th)/r(t), 0, 0, 0, 12*(1 - mu**2)/(E*(h**3)*r(t))],              #4
            [E*h/r(t) * ((np.cos(th))**2 + (self.k*h*np.sin(th))**2 / (6*(1 + mu)*(r(t)**2))), self.k*E*h/r(t) * np.cos(th),                            #5
                E*h/r(t) * np.sin(th)*np.cos(th) * (1 - (self.k*h)**2 / (6*(1 + mu)*(r(t)**2))),
                -(self.k**2)*E*(h**3) / (6*(1 + mu)*(r(t)**2)) * np.sin(th), mu/r(t) * np.cos(th), -self.k/r(t), -1/R1, 0,],
            [E*h/r(t) * self.k*np.cos(th), (self.k**2)*E*h/r(t), E*h/r(t) * self.k*np.sin(th) * (1 + ((self.k*h)**2) / (12 * r(t)**2)),                 #6
                E*(h**3) / (12*(r(t)**2)) * self.k*np.sin(th)*np.cos(th), mu*self.k/r(t), -np.cos(th)/r(t), 0, mu*self.k / (r(t)**2) * np.sin(th)],
            [E*h/r(t) * np.sin(th)*np.cos(th) * (1 - (self.k*h)**2 / (6*(1 + mu)*(r(t)**2))),                                                           #7
                E*h/r(t) * self.k*np.sin(th) * (1 + ((self.k*h)**2) / (12*(r(t)**2))), E*h/r(t) * (np.sin(th)**2 + (self.k**4)*(h**2)/(12*(r(t)**2))),
                (3+mu)/(1+mu) * E*(h**3)/(12*(r(t)**2)) * (self.k**2) * np.cos(th), 1/R1 + mu*np.sin(th)/r(t), 0, 0, mu*(self.k/r(t))**2],
            [-(self.k**2)*E*(h**3) / (6*(1 + mu)*(r(t)**2)) * np.sin(th), E*(h**3) / (12*(r(t)**2)) * self.k*np.sin(th)*np.cos(th),                     #8
                (3+mu)/(1+mu) * E*(h**3)/(12*(r(t)**2)) * (self.k**2) * np.cos(th),
                E*(h**3) / (12*r(t)) * (np.cos(th)**2 + 2*(self.k**2)/(1+mu)), 0, 0, 1, mu/r(t) * np.cos(th)],
            ])
        return ret_matr

    def load_coeff(self, t):
        # Коэффициенты разложения в ряд нагрузки
        # !!! Если 0 - поставь 0, не пропускай
        coeff_list = [-g*r(t)*ro/pi, -1./2.*g*r(t)*ro, -2./3.*g*r(t)*ro/pi, 0., 2./15.*g*r(t)*ro/pi]
        return coeff_list[self.k]

    def g_vec(self, t):
        return self.load_coeff(t) * np.array([0., 0., 0., 0., 0., 0., -r(t), 0.])

    # Специальная штука, которую передаем интегратору
    # Оставь как есть
    def F(self, t, y):
        return (self.Fmatr(t).dot(y))

    def F_plus_g(self, t, y):
        return (self.Fmatr(t).dot(y)) + self.g_vec(t)
###########



###########
# Даешь данные задачи
pi = 3.14159265
h = 0.002
L = 0.400
r1 = 0.200
r2 = 0.400
ro = 1000
mu = 0.3
R1 = 1e20 # Это такая бесконечность
E = 2e11
g = 9.81
num_of_harmonics = 5
###########
# Параметры
s0 = 0                                              # Это начало интервала интегрирования
smax = L / np.sin(pi/2 - np.arctan((r2-r1) / L))    # Это конец интервала интегрирования
parts = 20                                          # Это на сколько участков делим
steps = 2000                                        # Это сколько шагов интегрирования на участок
# Начальные условия. Которые не знаем - пиши как float('nan')
y_begin = [0., 0., 0., 0., float('nan'), float('nan'), float('nan'), float('nan')]
y_end = [0., 0., 0., 0., float('nan'), float('nan'), float('nan'), float('nan')]


###########

print('Количество гармоник:\n', num_of_harmonics)
print('Параметры установлены\n')
for i in range(num_of_harmonics):
    k = i
    print('Текущая гармоника:', k)
    eq_class = Diff_eq(k)
    print('Коэффициент нагрузки:', eq_class.load_coeff(smax))
    if (eq_class.load_coeff(smax) != 0.):
        yk_func_res = godunov_orthogonalization_solve(eq_class.F, eq_class.F_plus_g, y_begin, y_end, s0, smax, parts, steps)
        file_name = r'out\harmonic' + str(k) + '.csv'
        print(file_name)
        yk_func_res.to_csv(file_name, index = False)
