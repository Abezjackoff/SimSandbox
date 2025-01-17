import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.integrate import odeint
from scipy import optimize
import cvxpy as cvx

from dynamics.mech_sys_plant import PlantMechanicsModel
from control.mpc import MPController
from control.cart_pole import CartPolePlant


def blur_img(img):
    top = img[:-2, 1:-1]
    left = img[1:-1, :-2]
    center = img[1:-1, 1:-1]
    bottom = img[2:, 1:-1]
    right = img[1:-1, 2:]
    return (top + left + center + bottom + right) / 5

def do_blurring(name: str):
    img = plt.imread(name)

    res = timeit.timeit(lambda: blur_img(img), number=100)
    print(res / 100)

    blurred = blur_img(img)
    for _ in range(10):
        blurred = blur_img(blurred)

    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(blurred)

    plt.show()


def get_motion_solution(u, dT, n):
    x = np.zeros(n+1)
    v = np.zeros(n+1)
    for i in range(1, n+1):
        v[i] = v[i-1] + u[i-1, 0] * dT
        x[i] = x[i-1] + (v[i-1] + v[i]) * dT / 2
    return x, v


class PointMassPlant(PlantMechanicsModel):
    def __init__(self):
        super().__init__()
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1]])

    def build_system_dynamics(self):
        self.n_x = 2
        self.n_u = 1
        def ode_rhs(x, t, r, p=None):
            return self.A @ x + self.B @ r(x, t)

        self.rhs = ode_rhs
        return ode_rhs

    def get_lde_matrices(self, x, u):
        return self.A, self.B


class LinearCartPole:
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


class PointMassMPC(MPController):

    def __init__(self, a, plant: PlantMechanicsModel, y0, dT, finT, predT, ctrlT):
        super().__init__(plant, y0, dT, finT, predT, ctrlT)
        self.a = a
        self.toc_w = a * self.time

    def calc_cost(self, y):
        x = y[:, 0]
        v = y[:, 1]

        # J = 0.5 * np.sum((x-1)**2 + self.a * v**2)
        # J = np.sum(self.a * np.sqrt((x-1)**2 + v**2))
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


class TimeOptimalMPC(MPController):

    def __init__(self, a, plant: PlantMechanicsModel, y0, dT, finT, predT, ctrlT, stride):
        super().__init__(plant, y0, dT, finT, predT, ctrlT, stride)
        self.a = a
        self.toc_w = a * self.time**2

    def calc_cost(self, y):
        x = y[:, 0]
        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]

        J = np.sum(self.toc_w * np.sqrt((x - 25)**2 + theta**2 + v**2 + omega**2))
        return J

    def calc_grad(self, y):
        x = y[:, 0]
        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]

        J = np.sqrt((x - 25)**2 + theta**2 + v**2 + omega**2)
        dJ_dx = self.toc_w * (x - 25) / J
        dJ_dq = self.toc_w * theta / J
        dJ_dv = self.toc_w * v / J
        dJ_dw = self.toc_w * omega / J

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gq = G[1, 0]
        Gv = G[2, 0]
        Gw = G[3, 0]

        DJ = dJ_dx @ Gx + dJ_dq @ Gq + dJ_dv @ Gv + dJ_dw @ Gw
        return DJ

    def calc_hess(self, y):
        x = y[:, 0]
        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]

        J = np.sqrt((x - 25) ** 2 + theta ** 2 + v ** 2 + omega ** 2)
        ddJ_dxdx = self.toc_w * (1 - (x - 25)**2 / J**2) / J
        ddJ_dqdq = self.toc_w * (1 - theta**2 / J**2) / J
        ddJ_dvdv = self.toc_w * (1 - v**2 / J**2) / J
        ddJ_dwdw = self.toc_w * (1 - omega**2 / J**2) / J

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gq = G[1, 0]
        Gv = G[2, 0]
        Gw = G[3, 0]

        DDJ = Gx.T @ (ddJ_dxdx * Gx.T).T + Gq.T @ (ddJ_dqdq * Gq.T).T \
            + Gv.T @ (ddJ_dvdv * Gv.T).T + Gw.T @ (ddJ_dwdw * Gw.T).T
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
        # res = optimize.minimize(self.J_and_DJ, u, method='trust-constr', jac=True, hess=self.J_hess, bounds=bounds,
        #                         options={'verbose': 1, 'gtol': 1e-3})
        res = optimize.minimize(self.get_J_and_DJ, u, jac=True, bounds=bounds,
                                options={'gtol': 1e-3})

        u = res.x
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))


class TimeOptimalController:
    def __init__(self, plant, xT, F_max):
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

        res = optimize.fsolve(func, np.array([1e-3]), args=(a2,))
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


if __name__ == '__main__':

    # do_blurring('resources/starbucks-logo.png')

    dT = 0.5
    dT2 = dT*dT
    a = 5
    # A = np.array([[14*dT2+a, 8*dT2+a, 3*dT2+a, a],
    #               [8*dT2+a, 5*dT2+a, 2*dT2+a, a],
    #               [3*dT2+a, 2*dT2+a, dT2+a, a],
    #               [a, a, a, a]])
    # B = np.array([[6], [3], [1], [0]])
    # X = solve(A, B, assume_a='sym')
    # print(X)

    # Point mass
    #
    # n_steps = 80
    # u = np.zeros(n_steps)
    # x0 = np.zeros(2)
    #
    # plant = PointMassPlant()
    # plant.build_system_dynamics()
    # mpc = PointMassMPC(a, plant, x0, dT, n_steps * dT, 0.25 * n_steps * dT, 0.25 * n_steps * dT)
    # mpc.set_u_bounds([-2e-2, 2e-2])
    # t, y, u = mpc.run()
    #
    # # x, v = get_motion_solution(mpc.u, dT, n_steps)
    # # y = odeint(plant.rhs, x0, mpc.time, args=(mpc.get_u_lut,))
    # plt.figure('Response')
    # plt.plot(t, y[:, 0])
    # plt.plot(t, y[:, 1])
    # plt.figure('Control')
    # plt.plot(t, u[:, 0])
    # plt.show()

    # Time-optimal control of cart-pendulum system
    #
    x0 = np.zeros(4)
    x0[0] = 0
    x0[1] = 0
    t_step = 0.1
    xT = 25
    F_max = 80.

    act_time = 5.
    time = np.arange(0, 2*act_time + t_step, t_step)

    plant = LinearCartPole(5, 5, 3, 9.81)
    nlplant = CartPolePlant()
    nlplant.build_system_dynamics()
    nlplant.set_system_parameters(np.array([5, 5, 3, 9.81]))

    mpc = TimeOptimalMPC(a, nlplant, x0, t_step, 2*act_time, act_time, act_time, 20)
    mpc.set_u_bounds([-F_max], [F_max])
    t, y, u = mpc.run()

    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(t, y[:, 0], label='mpc')
    plt.xlabel('t, s')
    plt.ylabel('x, m')
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(t, y[:, 1], label='mpc')
    plt.xlabel('t, s')
    plt.ylabel(r'$ \theta $, rad')

    plt.figure()
    plt.grid()
    plt.plot(t, u)

    #
    # spec_ctrl = lambda x, t: [F_max] if t < act_time / 2 else [-F_max] if t < act_time else [0.]
    #
    # toc = TimeOptimalController(Plant, xT, F_max)
    # toc.calc_switch_points()
    # print(f'TOC: a = {np.sqrt(toc.a2)}')
    # print(f'TOC: phi = {np.sqrt(toc.k2) * (toc.T2 - toc.T1)}')

    # Phase portraits
    #
    # plt.figure()
    # a = np.linspace(0.001, 0.01, 10)
    # a = np.append(a, np.linspace(0.01, 20, 210))
    # x = [toc.get_x_root(a_**2) for a_ in a]
    # plt.grid()
    # plt.plot(a, x)
    # plt.xlabel('a, rad')
    # plt.ylabel(r'$ \phi^*$, rad')

    # t = np.linspace(0,2*np.pi, 100)
    # plt.plot(t, t - np.pi/2 * np.sin(t))

    # theta = np.linspace(-np.pi/2, np.pi/2, 200)
    # omega = np.sqrt(toc.k2) * theta
    # Theta, Omega = np.meshgrid(theta, omega)
    # DT_Pos = np.zeros_like(Theta)
    # DT_Neg = np.zeros_like(Theta)
    # DO_Pos = np.zeros_like(Omega)
    # DO_Neg = np.zeros_like(Omega)
    # for i in range(len(omega)):
    #     for j in range(len(theta)):
    #         states = np.array([0, Theta[i, j], 0, Omega[i, j]])
    #         derivs = Plant.rhs(states, 0, lambda x, t: [F_max])
    #         DT_Pos[i, j] = derivs[1]
    #         DO_Pos[i, j] = derivs[3]
    #         derivs = Plant.rhs(states, 0, lambda x, t: [-F_max])
    #         DT_Neg[i, j] = derivs[1]
    #         DO_Neg[i, j] = derivs[3]
    #
    # fig = plt.figure()
    # plt.subplot(1, 2, 2)
    # plt.grid()
    # plt.streamplot(Theta, Omega, DT_Pos, DO_Pos, color='C3', linewidth=0.5, density=0.6)
    # plt.streamplot(Theta, Omega, DT_Neg, DO_Neg, color='C0', linewidth=0.5, density=0.6)
    #
    # y = odeint(Plant.rhs, x0, time, args=(lambda x, t: [F_max],))
    # plt.plot(y[:,1], y[:,3], color='C3', label=r'$ u = u_{max} $')
    # y = odeint(Plant.rhs, x0, time, args=(lambda x, t: [-F_max],))
    # plt.plot(y[:, 1], y[:, 3], color='C0', label=r'$ u = -u_{max} $')
    # plt.scatter([0], [0], color='k')
    # plt.scatter([-F_max/(Plant.m + Plant.m1)/Plant.g], [0], color='C3')
    # plt.scatter([F_max/(Plant.m + Plant.m1)/Plant.g], [0], color='C0')
    # plt.xlim(-np.pi/2, np.pi/2)
    # plt.ylim(-4, 4)
    # plt.xlabel(r'$ \theta $, rad')
    # plt.ylabel(r'$ \omega $, rad/s')
    #
    # y = odeint(Plant.rhs, x0, time, args=(toc.get_ctrl,))
    # plt.plot(y[:, 1], y[:, 3], color='k', linestyle='dashed', label='3 switches')
    #
    # x = np.linspace(-10., 35., 200)
    # v = np.linspace(-20., 20., 200)
    # X, V = np.meshgrid(x, v)
    # DX_Pos = np.zeros_like(X)
    # DX_Neg = np.zeros_like(X)
    # DV_Pos = np.zeros_like(V)
    # DV_Neg = np.zeros_like(V)
    # for i in range(len(v)):
    #     for j in range(len(x)):
    #         states = np.array([X[i, j], 0, V[i, j], 0])
    #         derivs = Plant.rhs(states, 0, lambda x, t: [F_max])
    #         DX_Pos[i, j] = derivs[0]
    #         DV_Pos[i, j] = derivs[2]
    #         derivs = Plant.rhs(states, 0, lambda x, t: [-F_max])
    #         DX_Neg[i, j] = derivs[0]
    #         DV_Neg[i, j] = derivs[2]
    #
    # plt.subplot(1, 2, 1)
    # plt.grid()
    # plt.streamplot(X, V, DX_Pos, DV_Pos, color='C3', linewidth=0.5, density=0.6)
    # plt.streamplot(X, V, DX_Neg, DV_Neg, color='C0', linewidth=0.5, density=0.6)
    #
    # y = odeint(Plant.rhs, x0, time, args=(lambda x, t: [F_max],))
    # plt.plot(y[:, 0], y[:, 2], color='C3')
    # plt.plot(-y[:, 0] + 25, y[:, 2], color='C0')
    # plt.scatter([0, 25], [0, 0], color='k')
    # plt.xlim(-10, 35)
    # plt.ylim(-20, 20)
    # plt.xlabel('x, m')
    # plt.ylabel('v, m/s')
    #
    # y = odeint(Plant.rhs, x0, time, args=(toc.get_ctrl,))
    # plt.plot(y[:, 0], y[:, 2], color='k', linestyle='dashed')
    # plt.subplot(1, 2, 2)
    # plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)
    # plt.tight_layout()

    # Nonlinear system response
    # nlplant = CartPolePlant()
    # nlplant.build_system_dynamics()
    # nlplant.set_system_parameters(np.array([5, 5, 3, 9.81]))
    #
    # u = np.array([toc.get_ctrl(0, ti) for ti in time])
    # y = odeint(Plant.rhs, x0, time, args=(toc.get_ctrl, nlplant.const_dict))
    # fig = plt.figure('Response')
    # plt.subplot(1, 2, 1)
    # plt.grid()
    # plt.plot(time, y[:, 0], label='3 switches')
    # plt.xlabel('t, s')
    # plt.ylabel('x, m')
    # plt.subplot(1, 2, 2)
    # plt.grid()
    # plt.plot(time, y[:, 1], label='3 switches')
    # plt.xlabel('t, s')
    # plt.ylabel(r'$ \theta $, rad')

    # Intuitive 1-switch control
    #
    # spec_ctrl = lambda x, t: [F_max] if t < toc.T2 else [-F_max] if t < 2*toc.T2 else [0.]
    # y = odeint(Plant.rhs, x0, time, args=(spec_ctrl, nlplant.const_dict))
    # plt.subplot(1, 2, 1)
    # plt.plot(time, y[:, 0], linestyle='--', label='1 switch')
    # plt.subplot(1, 2, 2)
    # plt.plot(time, y[:, 1], linestyle='--', label='1 switch')
    #
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.tight_layout()

    # Hamiltonian
    #
    # H = toc.get_Hamiltonian(y, u, time)
    # plt.figure()
    # plt.grid()
    # plt.plot(time, H)
    # plt.xlabel('t, s')
    # plt.ylabel('H')


    #
    # plt.figure('Control')
    # # plt.plot(t, lqr_ctrl(y, 0)[0])
    # plt.plot(time, u[:, 0])
    #
    # plt.figure('Hamiltonian')
    # plt.plot(time, )
    #
    plt.show()
