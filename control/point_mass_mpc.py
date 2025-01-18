import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy import optimize

from dynamics.point_mass import PointMassPlant, get_motion_solution
from control.mpc import MPController

class PointMassMPC(MPController):

    def __init__(self, a, plant, y0, dT, finT, predT, ctrlT):
        super().__init__(plant, y0, dT, finT, predT, ctrlT)
        self.a = a
        self.toc_w = self.time

    def calc_cost(self, y):
        x = y[:, 0]
        v = y[:, 1]

        # J = 0.5 * np.sum((x-1)**2 + self.a * v**2)
        J = np.sum(self.toc_w * np.sqrt((x - 1) ** 2 + v ** 2))
        return J

    def calc_grad(self, y):
        x = y[:, 0]
        v = y[:, 1]

        # dJ_dx = x - 1
        # dJ_dv = self.a * v
        J = np.sqrt((x-1)**2 + v**2)
        dJ_dx = self.toc_w * (x - 1) / J
        dJ_dv = self.toc_w * v / J

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gv = G[1, 0]

        DJ = dJ_dx @ Gx + dJ_dv @ Gv
        return DJ

    def calc_hess(self, y):
        x = y[:, 0]
        v = y[:, 1]

        # ddJ_dxdx = np.ones(x.shape)
        # ddJ_dvdv = self.a * np.ones(v.shape)
        ddJ_dxdx = self.toc_w
        ddJ_dvdv = self.toc_w

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gv = G[1, 0]

        DDJ = Gx.T @ (ddJ_dxdx * Gx.T).T + Gv.T @ (ddJ_dvdv * Gv.T).T
        return DDJ

    def init_u(self):
        u = self.u.flatten()
        J, DJ = self.get_J_and_DJ(u)
        print(J)
        DDJ = self.get_J_hess(u)
        # print(np.linalg.cond(DDJ))
        u = solve(DDJ, -DJ)
        print(self.get_J_and_DJ(u)[0])
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))

    def optimize_u(self):
        # self.init_u()
        u = self.u.flatten()

        if self.has_bounds:
            bounds = optimize.Bounds(self.u_min, self.u_max)
        else:
            bounds = None
        res = optimize.minimize(self.get_J_and_DJ, u, jac=True, bounds=bounds)

        u = res.x
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))


if __name__ == '__main__':
    dT = 0.5
    dT2 = dT*dT
    a = 5
    A = np.array([[14*dT2+a, 8*dT2+a, 3*dT2+a, a],
                  [8*dT2+a, 5*dT2+a, 2*dT2+a, a],
                  [3*dT2+a, 2*dT2+a, dT2+a, a],
                  [a, a, a, a]])
    B = np.array([[6], [3], [1], [0]])
    X = solve(A, B, assume_a='sym')
    print(X)


    n_steps = 80
    u = np.zeros(n_steps)
    x0 = np.zeros(2)

    plant = PointMassPlant(1)
    mpc = PointMassMPC(a, plant, x0, dT, n_steps * dT, 0.25 * n_steps * dT, 0.25 * n_steps * dT)
    mpc.set_u_bounds([-2e-2, 2e-2])
    t, y, u = mpc.run()

    # x, v = get_motion_solution(mpc.u, dT, n_steps)
    # y = odeint(plant.rhs, x0, mpc.time, args=(mpc.get_u_lut,))

    plt.figure('Response')
    plt.plot(t, y[:, 0])
    plt.plot(t, y[:, 1])
    plt.figure('Control')
    plt.plot(t, u[:, 0])
    plt.show()
