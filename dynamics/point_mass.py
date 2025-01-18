import numpy as np

class PointMassPlant:
    def __init__(self, m):
        self.m = m
        self.n_x = 2
        self.n_u = 1
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1/m]])

    def rhs(self, x, t, r, p=None):
        return self.A @ x + self.B @ r(x, t)

    def get_lde_matrices(self, x, u):
        return self.A, self.B

def get_motion_solution(u, dT, n):
    x = np.zeros(n+1)
    v = np.zeros(n+1)
    for i in range(1, n+1):
        v[i] = v[i-1] + u[i-1, 0] * dT
        x[i] = x[i-1] + (v[i-1] + v[i]) * dT / 2
    return x, v
