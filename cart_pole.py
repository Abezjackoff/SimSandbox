import matplotlib.pyplot as plt
import numpy as np
from pydy.codegen.ode_function_generators import generate_ode_function
from sympy import Symbol, symbols, trigsimp, lambdify
from sympy.physics.mechanics import Particle, KanesMethod
from sympy.physics.vector import dynamicsymbols, Point
from scipy.linalg import solve
from scipy import optimize
from pydy.viz.shapes import Cylinder, Sphere, Box
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene

from dyn_sys_plant import PlantMechanicsModel
from mpc import MPController


class CartPolePlant(PlantMechanicsModel):

    def __init__(self):
        super().__init__()

    def build_system_dynamics(self):

        x, theta = dynamicsymbols('x theta')
        v, omega = dynamicsymbols('v omega')
        self.coords = [x, theta]
        self.speeds = [v, omega]
        self.n_x = 4

        m, m1 = symbols('m m1')
        g, l1 = symbols('g l1')
        self.constants = [m, m1, l1, g]

        Fx = dynamicsymbols('Fx')
        self.specified = [Fx]
        self.n_u = 1

        self.add_frame('I')
        self.add_frame('P')
        self.frames['P'].orient(self.frames['I'], 'Axis', (theta, self.frames['I'].z))

        self.add_point('O')
        self.points['O'].set_vel(self.frames['I'], 0)
        self.add_point('C')
        self.points['C'].set_pos(self.points['O'], x * self.frames['I'].x)
        self.add_point('P')
        self.points['P'].set_pos(self.points['C'], -l1 * self.frames['P'].y)

        Cart = Particle('Cart', self.points['C'], m)
        Mass = Particle('Mass', self.points['P'], m1)

        kinematics = [v - x.diff(), omega - theta.diff()]


        cart_force = (self.points['C'], Fx * self.frames['I'].x)
        mass_force = (self.points['P'], -m1 * g * self.frames['I'].y)

        self.kane = KanesMethod(self.frames['I'], q_ind=self.coords, u_ind=self.speeds, kd_eqs=kinematics)
        bodies = [Cart, Mass]
        loads = [cart_force, mass_force]

        fr, frstar = self.kane.kanes_equations(bodies, loads)
        mass_matrix = trigsimp(self.kane.mass_matrix_full)
        forcing_vec = trigsimp(self.kane.forcing_full)
        # print(mass_matrix)
        # print(forcing_vec)

        right_hand_side = generate_ode_function(forcing_vec, self.coords, self.speeds, self.constants,
                                                mass_matrix=mass_matrix, specifieds=self.specified)

        self.rhs = right_hand_side
        return right_hand_side

    def get_lde_matrices(self, x, u):
        m = self.const_dict[Symbol('m')]
        m1 = self.const_dict[Symbol('m1')]
        g = self.const_dict[Symbol('g')]
        l1 = self.const_dict[Symbol('l1')]

        sin = np.sin(x[1])
        cos = np.cos(x[1])
        sin2 = np.sin(2*x[1])
        cos2 = np.cos(2*x[1])
        dnom = m + m1 - m1 * (1 + cos2) / 2

        A = np.zeros((4, 4))
        B = np.zeros((4, 1))

        A[0, 2] = 1
        A[1, 3] = 1
        A[2, 1] = -m1 * sin2 / dnom**2 * (u[0] + m1*g/l1 * sin2 / 2 + m1*l1 * x[3]**2 * sin) \
                + 1 / dnom * (m1*g/l1 * cos2 + m1*l1 * x[3]**2 * cos)
        A[2, 3] = 2 * m1 * l1 * x[3] * sin / dnom
        A[3, 1] = -m1 * sin2 / dnom**2 * (-u[0] * cos - (m + m1)*g/l1 * sin - m1*l1 * x[3]**2 * sin2 / 2) \
                + 1 / dnom * (u[0] * sin - (m + m1)*g/l1 * cos - m1*l1 * x[3]**2 * cos2)
        A[3, 3] = -m1 * l1 * x[3] * sin2 / dnom

        B[2, 0] = 1 / dnom
        B[3, 0] = -cos / dnom

        A[2, 1] = -m1 * sin2 / dnom**2 * (u[0] + m1*g/l1 * sin2 / 2 + m1*l1 * x[3]**2 * sin) \
                + 1 / dnom * (m1*g/l1 * cos2 + m1*l1 * x[3]**2 * cos)

        # A = np.array([[ 0.,     0.,     1.,     0.   ],
        #               [ 0.,     0.,     0.,     1.   ],
        #               [ 0.,     4.905,  0.,     0.   ],
        #               [ 0.,    14.715,  0.,     0.   ]])
        # B = np.array([[0. ],
        #               [0. ],
        #               [0.1],
        #               [0.1]])

        return A, B


class CartPoleMPC(MPController):

    def __init__(self, plant, y0, dT, finT, predT, ctrlT):
        super().__init__(plant, y0, dT, finT, predT, ctrlT)

    def get_cost(self, y):
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

    def get_grad(self, y):
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

    def get_hess(self, y):
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

        for i in range(self.n_ctrl):
            t = self.time[i]
            if 0 <= t < 0.15:
                u[i] = 300
            elif 1 <= t < 1.15:
                u[i] = -300
            else:
                u[i] = 0

        # bounds = optimize.Bounds(-300 * np.ones(self.n_ctrl), 300 * np.ones(self.n_ctrl))
        # res = optimize.direct(self.J_func, bounds, locally_biased=False, maxiter=100, callback=self.print_progress)
        # u = res.x
        # print(res.fun)

        self.u = u.reshape((self.n_ctrl, self.plant.n_u))

    def optimize_u(self):
        # self.init_u()
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


def pydy_viz(plant, y):
    cart_shape = Box(color='black', width=0.4, height=0.2, depth=0.2)
    mass_shape = Sphere(color='black', radius=0.1)
    rod_shape = Cylinder(color='blue', radius=0.05, length=plant.constants[2])

    R = Point('R')
    R.set_pos(plant.points['C'], -plant.constants[2]/2 * plant.frames['P'].y)

    cart_viz_frame = VisualizationFrame(plant.frames['I'], plant.points['C'], cart_shape)
    mass_viz_frame = VisualizationFrame(plant.frames['I'], plant.points['P'], mass_shape)
    rod_viz_frame = VisualizationFrame(plant.frames['P'], R, rod_shape)

    scene = Scene(plant.frames['I'], plant.points['O'])
    scene.visualization_frames = [cart_viz_frame, mass_viz_frame, rod_viz_frame]

    scene.states_symbols = plant.coords + plant.speeds
    scene.constants = plant.const_dict
    scene.states_trajectories = y
    scene.display()


if __name__ == '__main__':

    plant = CartPolePlant()
    rhs = plant.build_system_dynamics()
    # help(rhs)
    plant.set_system_parameters(np.array([10, 5, 1, 9.81]))

    # Set IC and simulation time
    x0 = np.zeros(4)
    x0[0] = 2
    x0[1] = np.pi - np.deg2rad(45)
    t_step = 0.05
    act_time = 10.
    time = np.arange(0, 2*act_time + t_step, t_step)

    # Specified control
    F_max = 80.
    spec_ctrl = lambda x, t: [F_max] if t < act_time/2 else [-F_max] if t < act_time else [0.]

    # LQR control
    x_targ = np.zeros(4)
    x_targ[0] = 0
    x_targ[1] = np.pi

    Q = np.diag([5, 20, 1, 10])
    R = np.diag([0.01])
    K = plant.get_lqr_gains(x_targ, Q, R)
    lqr_ctrl = lambda x, t: K @ (x_targ - x).T

    # MPC control
    mpc = CartPoleMPC(plant, x0, t_step, act_time, 0.1*act_time, 0.1*act_time)
    t, y, u = mpc.run()

    # Mixed MPC and LQR control
    # def mixed_control(x, t):
    #     if np.cos(x[1]) <= -0.5:
    #         return lqr_ctrl(x, t)
    #     else:
    #         return mpc.get_u_lut(x, t)

    # Get control response
    # y = odeint(rhs, x0, time, args=(mixed_control, plant.const_dict))
    # t = time
    # print(mpc.get_cost(y))
    # mpc.set_u_lut(mixed_control, y)

    pydy_viz(plant, y)

    P_x = plant.points['P'].pos_from(plant.points['O']).dot(plant.frames['I'].x)
    pos1_func = lambdify((plant.coords[0], plant.coords[1], plant.constants[2]), P_x, 'numpy')

    plt.figure('Response')
    plt.plot(t, y[:, 0])
    plt.plot(t, pos1_func(y[:,0], y[:,1], plant.const_dict[plant.constants[2]]))
    plt.xlabel('Time [s]')
    plt.ylabel('X [m]')
    plt.legend(['Cart', 'Mass'])

    plt.figure('Control')
    # plt.plot(t, lqr_ctrl(y, 0)[0])
    plt.plot(t, u[:, 0])

    plt.show()
