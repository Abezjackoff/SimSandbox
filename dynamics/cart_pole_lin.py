import numpy as np

class CartPoleLinear:
    def __init__(self, m, m1, l1, g):
        self.const_dict = dict()
        self.m = m
        self.m1 = m1
        self.l1 = l1
        self.g = g
        self.n_x = 4
        self.n_u = 1
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, self.m1 / self.m * self.g, 0, 0],
                           [0, -(self.m + self.m1) / self.m / self.l1 * self.g, 0, 0]])
        self.B = np.array([[0],
                           [0],
                           [1 / self.m],
                           [-1 / self.m / self.l1]])

    def rhs(self, x, t, r, p=None):
        return self.A @ x + self.B @ r(x, t)

    def get_lde_matrices(self, x, u):
        return self.A, self.B
