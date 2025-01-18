import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import lambdify
from sympy.physics.vector import Point
from pydy.viz.shapes import Cylinder, Sphere, Box
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene

from dynamics.cart_pole_plant import CartPolePlant
from control.cart_pole_mpc import CartPoleMPC
from control.cart_pole_toc import TimeOptimalController


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


def swing_up(plant, x0, t_step, act_time):
    # LQR control
    # x_targ = np.zeros(4)
    # x_targ[0] = 0
    # x_targ[1] = np.pi
    #
    # Q = np.diag([5, 20, 1, 10])
    # R = np.diag([0.01])
    # K = plant.get_lqr_gains(x_targ, Q, R)
    # lqr_ctrl = lambda x, t: K @ (x_targ - x).T

    # MPC control
    mpc = CartPoleMPC(plant, x0, t_step, act_time, 0.5*act_time, 0.5*act_time)
    mpc.set_u_bounds([-300], [300])
    mpc.init_u()
    t, y, u = mpc.run()

    # Mixed MPC and LQR control
    # def mixed_control(x, t):
    #     if np.cos(x[1]) <= -0.5:
    #         return lqr_ctrl(x, t)
    #     else:
    #         return mpc.get_u_lut(x, t)

    # Get control response
    # y = odeint(rhs, x0, time, args=(spec_ctrl, plant.const_dict))
    # t = time
    # u = np.array([spec_ctrl(0, ti) for ti in t])
    # print(mpc.get_cost(y))
    # mpc.set_u_lut(mixed_control, y)

    return t, y, u


if __name__ == '__main__':

    # Build nonlinear dynamic system
    plant = CartPolePlant()
    rhs = plant.build_system_dynamics()
    # help(rhs)
    # plant.set_system_parameters(np.array([10, 5, 1, 9.81]))
    plant.set_system_parameters(np.array([5, 5, 3, 9.81]))

    # Set IC and simulation time
    x0 = np.zeros(4)
    x0[0] = 0
    x0[1] = 0
    F_max = 80.

    t_step = 0.05
    act_time = 5.
    time = np.arange(0, 2*act_time + t_step, t_step)

    # Manual specified control
    T1 = 1.7
    T2 = 2 * T1
    spec_ctrl = lambda x, t: [F_max] if t < T1 else [-F_max] if t < T2 else [0.]
    # t = time
    # y = odeint(rhs, x0, time, args=(spec_ctrl, plant.const_dict))
    # u = [spec_ctrl(0, t) for t in time]

    toc = TimeOptimalController(plant, 25., F_max, use_dict=True)
    toc.calc_switch_points()
    t = time
    y = odeint(rhs, x0, time, args=(toc.get_ctrl, plant.const_dict))
    u = [toc.get_ctrl(0, t) for t in time]

    # Swing up control
    # t, y, u = swing_up(plant, x0, t_step, act_time)

    pydy_viz(plant, y[::2])

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
