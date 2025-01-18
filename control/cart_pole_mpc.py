import numpy as np
from scipy import optimize
from sympy import Symbol
from control.mpc import MPController

class CartPoleMPC(MPController):

    def __init__(self, plant, y0, dT, finT, predT, ctrlT):
        super().__init__(plant, y0, dT, finT, predT, ctrlT)

    def calc_cost(self, y):
        m = self.plant.const_dict[Symbol('m')]
        m1 = self.plant.const_dict[Symbol('m1')]
        g = self.plant.const_dict[Symbol('g')]
        l1 = self.plant.const_dict[Symbol('l1')]

        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]
        cos = np.cos(theta)
        sin = np.sin(theta)

        E_pot = -m1 * g * l1 * np.sum(cos)
        E_kin = 0
        E_kin += 0.5 * m * np.sum(v**2)
        E_kin += 0.5 * m1 * np.sum((v + l1 * omega * cos)**2)
        E_kin += 0.5 * m1 * np.sum((l1 * omega * sin)**2)
        J = E_kin - E_pot

        # E_pot = m1 * g * l1 * np.sum(0.5 * theta**2 - 1)
        # E_kin = 0.5 * m * np.sum(v**2) + 0.5 * m1 * np.sum((v**2 + 2 * l1 * v * omega + l1**2 * omega**2))
        # J = E_kin + E_pot

        # J = 0.5 * (5*np.sum(y[:, 0]**2) + 20*np.sum((theta - np.pi)**2) + 1*np.sum(v**2) + 10*np.sum(omega**2) \
        #   + 0.01 * np.sum(self.u[:, 0]**2) + 0.01 * (self.n_pred - self.n_ctrl - 1) * self.u[-1, 0]**2)
        return J

    def calc_grad(self, y):
        m = self.plant.const_dict[Symbol('m')]
        m1 = self.plant.const_dict[Symbol('m1')]
        g = self.plant.const_dict[Symbol('g')]
        l1 = self.plant.const_dict[Symbol('l1')]

        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]
        cos = np.cos(theta)
        sin = np.sin(theta)

        dJ_dq = -m1 * l1 * (g + v * omega) * sin
        dJ_dv = (m + m1) * v + m1 * l1 * omega * cos
        dJ_dw = m1 * l1 * (v * cos + l1 * omega)

        # dJ_dq = m1 * l1 * g * theta
        # dJ_dv = (m + m1) * v + m1 * l1 * omega
        # dJ_dw = m1 * l1 * v + m1 * l1**2 * omega

        # dJ_dx = 5 * y[:, 0]
        # dJ_dq = 20 * (theta - np.pi)
        # dJ_dv = 1 * v
        # dJ_dw = 10 * omega
        # dJ_du = 0.01 * self.u[:, 0]
        # dJ_du[-1] += 0.01 * (self.n_pred - self.n_ctrl - 1) * self.u[-1, 0]

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gq = G[1, 0]
        Gv = G[2, 0]
        Gw = G[3, 0]

        DJ = dJ_dq @ Gq + dJ_dv @ Gv + dJ_dw @ Gw

        # DJ = dJ_dx @ Gx + dJ_dq @ Gq + dJ_dv @ Gv + dJ_dw @ Gw + dJ_du
        return DJ

    def calc_hess(self, y):
        m = self.plant.const_dict[Symbol('m')]
        m1 = self.plant.const_dict[Symbol('m1')]
        g = self.plant.const_dict[Symbol('g')]
        l1 = self.plant.const_dict[Symbol('l1')]

        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]
        cos = np.cos(theta)
        sin = np.sin(theta)

        ddJ_dqdq = -m1 * l1 * (g + v * omega) * cos
        ddJ_dqdv = -m1 * l1 * omega * sin
        ddJ_dqdw = -m1 * l1 * v * sin
        ddJ_dvdv = (m + m1) * np.ones(v.shape)
        ddJ_dvdw = m1 * l1 * cos
        ddJ_dwdw = m1 * l1**2 * np.ones(omega.shape)

        # ddJ_dxdx = 5 * np.ones(self.n_pred)
        # ddJ_dqdq = 20 * np.ones(self.n_pred)
        # ddJ_dvdv = 1 * np.ones(self.n_pred)
        # ddJ_dwdw = 10 * np.ones(self.n_pred)
        # ddJ_dudu = 0.01 * np.ones(self.n_ctrl)
        # ddJ_dudu[-1] += 0.01 * (self.n_pred - self.n_ctrl - 1)

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gq = G[1, 0]
        Gv = G[2, 0]
        Gw = G[3, 0]

        DDJ = Gq.T @ ((ddJ_dqdq * Gq.T).T + (ddJ_dqdv * Gv.T).T + (ddJ_dqdw * Gw.T).T) \
            + Gv.T @ ((ddJ_dqdv * Gq.T).T + (ddJ_dvdv * Gv.T).T + (ddJ_dvdw * Gw.T).T) \
            + Gw.T @ ((ddJ_dqdw * Gq.T).T + (ddJ_dvdw * Gv.T).T + (ddJ_dwdw * Gw.T).T)

        # DDJ = Gx.T @ (ddJ_dxdx * Gx.T).T + Gq.T @ (ddJ_dqdq * Gq.T).T \
        #     + Gv.T @ (ddJ_dvdv * Gv.T).T + Gw.T @ (ddJ_dwdw * Gw.T).T + np.diag(ddJ_dudu)
        return DDJ

    def init_u(self):
        u = self.u.flatten()

        # J, DJ = self.J_and_DJ(u)
        # # print(J)
        # for i in range(1):
        #     DDJ = self.J_hess(u)
        #     # print(np.linalg.cond(DDJ))
        #     u = solve(DDJ, -DJ)
        #     J, DJ = self.J_and_DJ(u)
        #     print(J)

        if self.has_bounds:
            bounds = optimize.Bounds(self.u_min, self.u_max)
            for i in range(self.n_ctrl):
                t = self.time[i]
                if 0 <= t < 0.15:
                    u[i] = self.u_max[0]
                elif 1.5 <= t < 1.65:
                    u[i] = self.u_min[0]
                else:
                    u[i] = 0
        else:
            bounds = None

        # res = optimize.direct(self.J_func, bounds, locally_biased=False, maxiter=100, callback=self.print_progress)
        # u = res.x
        # print(res.fun)

        self.u = u.reshape((self.n_ctrl, self.plant.n_u))

    def optimize_u(self):
        u = self.u.flatten()

        if self.has_bounds:
            bounds = optimize.Bounds(self.u_min, self.u_max)
        else:
            bounds = None
        res = optimize.minimize(self.get_J_and_DJ, u, method='trust-constr', jac=True, hess=self.get_J_hess, bounds=bounds,
                                options={'verbose': 1, 'gtol': 1e-3})
        # res = optimize.minimize(self.J_and_DJ, u, method='trust-ncg', jac=True, hess=self.J_hess,
        #                         options={'disp': True, 'gtol': 1e-3})
        # res = optimize.minimize(self.J_and_DJ, u, method='Newton-CG', jac=True, hess=self.J_hess,
        #                         options={'disp': True, 'gtol': 1e-3})
        # res = optimize.minimize(self.J_and_DJ, u, method='BFGS', jac=True,
        #                         options={'disp': True, 'gtol': 1e-3})
        # res = optimize.minimize(self.J_func, u, method='nelder-mead', callback=self.print_progress,
        #                         options={'disp': True, 'maxiter': 1000})
        u = res.x
        print(res.fun)

        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
