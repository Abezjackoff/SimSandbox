import numpy as np
from scipy.optimize import fsolve
from sympy import Symbol

class TimeOptimalController:
    def __init__(self, plant, xT, F_max, use_dict=False):
        if use_dict:
            self.m = plant.const_dict[Symbol('m')]
            self.m1 = plant.const_dict[Symbol('m1')]
            self.l1 = plant.const_dict[Symbol('l1')]
            self.g = plant.const_dict[Symbol('g')]
        else:
            self.m = plant.m
            self.m1 = plant.m1
            self.l1 = plant.l1
            self.g = plant.g
        self.xT = xT
        self.F_max = F_max
        self.T1 = 0
        self.T2 = 0
        self.T3 = 0
        self.T4 = 0
        self.k2 = 0
        self.a2 = 0

    def calc_switch_points(self):
        self.k2 = (self.m + self.m1) / self.m * self.g / self.l1
        self.a2 = self.k2 * self.xT * (self.m + self.m1) / self.F_max

        tau = self.get_x_root(self.a2) / np.sqrt(self.k2)
        self.T2 = np.sqrt(self.a2 / self.k2 + 2 * tau**2)
        self.T1 = self.T2 - tau
        self.T3 = self.T2 + tau
        self.T4 = 2 * self.T2

    @staticmethod
    def get_x_root(a2):
        def func(x, a2):
            return np.cos(np.sqrt(a2 + 2 * x**2)) - 2 * np.cos(x) + 1

        res = fsolve(func, np.array([1e-3]), args=(a2,))
        return np.abs(res[0])

    def get_Hamiltonian(self, y, u, t):
        k = np.sqrt(self.k2)
        phi = k * (self.T2 - self.T1)
        A = -1
        C = phi / np.sin(phi)
        B = A * C
        c1 = A * (self.m + self.m1) / self.m * k
        c2 = A * (self.m + self.m1) / self.m * k * self.T2
        c3 = B * self.l1 * k * np.cos(k * self.T2)
        c4 = B * self.l1 * k * np.sin(k * self.T2)

        lm1 = c1 * np.ones_like(t)
        lm2 = c3 * np.cos(k * t) + c4 * np.sin(k * t) + self.m1 * self.g / self.m / k ** 2 * lm1
        lm3 = -c1 * t + c2
        lm4 = -c3 / k * np.sin(k * t) + c4 / k * np.cos(k * t) + self.m1 * self.g / self.m / k ** 2 * lm3

        H = 1 + y[:,2] * lm1 + y[:,3] * lm2 + \
             (self.m1*self.g*y[:, 1] + u[:, 0]) * lm3/self.m - \
             ((self.m+self.m1)*self.g*y[:, 1] + u[:, 0]) * lm4/self.m/self.l1
        return H


    def get_ctrl(self, x, t):
        if 0 <= t < self.T1:
            u = 1
        elif self.T1 <= t < self.T2:
            u = -1
        elif self.T2 <= t < self.T3:
            u = 1
        elif self.T3 <= t < self.T4:
            u = -1
        else:
            u = 0
        return [u * self.F_max]
