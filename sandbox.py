import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.integrate import odeint
from scipy import optimize

from dynamics.cart_pole_lin import CartPoleLinear
from control.mpc import MPController
from control.cart_pole import CartPolePlant
from control.cart_pole_toc import TimeOptimalController


class TimeOptimalMPC(MPController):

    def __init__(self, plant, y0, dT, finT, predT, ctrlT, stride):
        super().__init__(plant, y0, dT, finT, predT, ctrlT, stride)
        self.toc_w = self.time

    def calc_cost(self, y):
        x = y[:, 0]
        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]

        J = np.sum(self.toc_w * np.sqrt((x - 25)**2 + 5 * theta**2 + v**2 + 5 * omega**2))
        return J

    def calc_grad(self, y):
        x = y[:, 0]
        theta = y[:, 1]
        v = y[:, 2]
        omega = y[:, 3]

        J = np.sqrt((x - 25)**2 + 5 * theta**2 + v**2 + 5 * omega**2)
        dJ_dx = self.toc_w * (x - 25) / J
        dJ_dq = self.toc_w * 5 * theta / J
        dJ_dv = self.toc_w * v / J
        dJ_dw = self.toc_w * 5 * omega / J

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

        J = np.sqrt((x - 25) ** 2 + 5 * theta ** 2 + v ** 2 + 5 * omega ** 2)
        ddJ_dxdx = self.toc_w * (1 - (x - 25)**2 / J**2) / J
        ddJ_dqdq = self.toc_w * 5 * (1 - 5 * theta**2 / J**2) / J
        ddJ_dvdv = self.toc_w * (1 - v**2 / J**2) / J
        ddJ_dwdw = self.toc_w * 5 * (1 - 5 * omega**2 / J**2) / J

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
        res = optimize.minimize(self.get_J_and_DJ, u, method='trust-constr', jac=True, hess=self.get_J_hess, bounds=bounds,
                                options={'verbose': 1, 'gtol': 1e-3})
        # res = optimize.minimize(self.get_J_and_DJ, u, jac=True, bounds=bounds,
        #                         options={'gtol': 1e-3})

        u = res.x
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))


if __name__ == '__main__':

    # Time-optimal control of cart-pendulum system
    #
    x0 = np.zeros(4)
    x0[0] = 0
    x0[1] = 0
    xT = 25
    F_max = 80.

    t_step = 0.05
    act_time = 5.
    time = np.arange(0, 2*act_time + t_step, t_step)

    plant = CartPoleLinear(5, 5, 3, 9.81)
    nlplant = CartPolePlant()
    nlplant.build_system_dynamics()
    nlplant.set_system_parameters(np.array([5, 5, 3, 9.81]))



    mpc = TimeOptimalMPC(nlplant, x0, t_step, 2*act_time, 0.7*act_time, 0.7*act_time, 5)
    mpc.set_u_bounds([-F_max], [F_max])
    t, y, u = mpc.run()
    np.save('t_nl', t)
    np.save('y_nl', y)
    np.save('u_nl', u)

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
