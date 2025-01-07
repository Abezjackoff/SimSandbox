import abc
import numpy as np
from scipy.integrate import odeint
from scipy import optimize

from dyn_sys_plant import PlantMechanicsModel


class MPController:

    def __init__(self, plant: PlantMechanicsModel, y0, dT, finT, predT, ctrlT):
        self.plant = plant
        self.y0 = y0
        self.dT = dT
        self.finT = finT
        self.time = np.arange(0, predT + dT, dT)
        self.n_pred = self.time.shape[0]
        self.n_ctrl = min(int(np.round(ctrlT / dT)), self.n_pred - 1)
        self.u = np.zeros((self.n_ctrl, self.plant.n_u))
        self.cost = 0
        self.iter = 0

    def get_u_lut(self, x, t):
        i = int(np.floor(t / self.dT))
        i = min(i, self.n_ctrl - 1)
        return self.u[i]

    def set_u_lut(self, func_u, y=None):
        for i in range(self.n_ctrl):
            if y is None:
                x = 0
            else:
                x = y[i]
            t = self.time[i]
            self.u[i] = func_u(x, t)

    def get_sensitivity(self, y, u):
        G = np.zeros((self.plant.n_x, self.plant.n_u, self.n_pred, self.n_ctrl))

        A_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_x))
        B_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_u))
        AA_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_x))
        AB_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_u))
        for i in range(0, self.n_pred-1):
            x = np.array(y[i])
            r = np.array(u[min(i, self.n_ctrl - 1)])
            A, B = self.plant.get_lde_matrices(x, r)
            A_trj[i] = A
            B_trj[i] = B
            # x = np.array(y[i])
            # A, B = plant.get_lde_matrices(x, r)
            # A_trj[i] += A
            # B_trj[i] += B
            # A_trj[i] *= 0.5
            # B_trj[i] *= 0.5
            AA_trj[i] = A_trj[i] @ A_trj[i]
            AB_trj[i] = A_trj[i] @ B_trj[i]

        for j in range(0, self.n_ctrl):
            for i in range(j+1, self.n_pred):
                if i == j+1:
                    z = B_trj[i-1] * self.dT + 0.5 * AB_trj[i-1] * self.dT ** 2
                else:
                    z = np.zeros((self.plant.n_x, self.plant.n_u))
                    for k in range(0, self.plant.n_x):
                        for l in range(0, self.plant.n_u):
                            z[k, l] = G[k, l][i-1, j]
                    z = (np.eye(self.plant.n_x) + A_trj[i-1] * self.dT + 0.5 * AA_trj[i-1] * self.dT**2) @ z

                for k in range(0, self.plant.n_x):
                    for l in range(0, self.plant.n_u):
                        G[k, l][i, j] = z[k, l]
        return G

    @abc.abstractmethod
    def get_cost(self, y):
        pass

    @abc.abstractmethod
    def get_grad(self, y):
        pass

    @abc.abstractmethod
    def get_hess(self, y):
        pass

    def J_and_DJ(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        J = self.get_cost(y)
        DJ = self.get_grad(y)
        self.cost = J
        return (J, DJ)

    def J_func(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        J = self.get_cost(y)
        self.cost = J
        return J

    def J_grad(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        DJ = self.get_grad(y)
        return DJ

    def J_hess(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        DDJ = self.get_hess(y)
        return DDJ

    def print_progress(self, x, e=None, context=None):
        self.iter += 1
        print(f'#{self.iter}   {self.cost}')

    def init_u(self):
        # u = self.u.flatten()

        # J, DJ = self.J_and_DJ(u)
        # print(J)
        # for i in range(2):
        #     DDJ = self.J_hess(u)
        #     print(np.linalg.cond(DDJ))
        #     u = solve(DDJ, -DJ)
        #     J, DJ = self.J_and_DJ(u)
        #     print(J)

        # bounds = optimize.Bounds(-300 * np.ones(self.n_ctrl), 300 * np.ones(self.n_ctrl))
        # res = optimize.direct(self.J_func, bounds, locally_biased=False, maxiter=100, callback=self.print_progress)
        # u = res.x
        # print(res.fun)

        # self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        pass

    def optimize_u(self):
        u = self.u.flatten()

        bounds = optimize.Bounds(-300 * np.ones(self.n_ctrl), 300 * np.ones(self.n_ctrl))
        res = optimize.minimize(self.J_and_DJ, u, method='trust-constr', jac=True, hess=self.J_hess, bounds=bounds,
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

    def run(self):
        t_full = np.array([])
        y_full = np.zeros((0, self.plant.n_x))
        u_full = np.zeros((0, self.plant.n_u))

        self.optimize_u()
        y = odeint(self.plant.rhs, self.y0, self.time[0:2], args=(self.get_u_lut, self.plant.const_dict))
        t_full = np.append(t_full, self.time[0])
        y_full = np.append(y_full, [self.y0], axis=0)
        u_full = np.append(u_full, [self.u[0]], axis=0)

        t_stop = self.finT
        while t_stop > self.dT:
            self.y0 = y[-1]
            self.u = np.roll(self.u, -1, axis=0)
            self.u[-1] = self.u[-2]

            self.optimize_u()
            y = odeint(self.plant.rhs, self.y0, self.time[0:2], args=(self.get_u_lut, self.plant.const_dict))
            t_full = np.append(t_full, t_full[-1] + self.dT)
            y_full = np.append(y_full, [self.y0], axis=0)
            u_full = np.append(u_full, [self.u[0]], axis=0)

            t_stop -= self.dT

        t_full = np.append(t_full, t_full[-1] + self.dT)
        y_full = np.append(y_full, [y[1]], axis=0)
        u_full = np.append(u_full, [u_full[-1]], axis=0)
        return t_full, y_full, u_full
