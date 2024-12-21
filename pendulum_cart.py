import matplotlib.pyplot as plt
import numpy as np
from pydy.codegen.ode_function_generators import generate_ode_function
from sympy import symbols, simplify, trigsimp, Matrix, matrix2numpy, latex, lambdify
from sympy.physics.mechanics import Particle, KanesMethod
from sympy.physics.vector import dynamicsymbols, ReferenceFrame, Point, dot, cross, init_vprinting
from scipy.integrate import odeint
from pydy.viz.shapes import Cylinder, Sphere, Box
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene

def build_motion_equations():


    return right_hand_side


if __name__ == '__main__':

    inertial_frame = ReferenceFrame('I')

    x, theta1, theta2 = dynamicsymbols('x theta1 theta2')
    v, omega1, omega2 = dynamicsymbols('v omega1 omega2')
    m, m1, m2 = symbols('m m1 m2')
    g, l1, l2 = symbols('g l1 l2')

    mass1_frame = ReferenceFrame('P1')
    mass1_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))

    mass2_frame = ReferenceFrame('P2')
    mass2_frame.orient(inertial_frame, 'Axis', (theta2, inertial_frame.z))

    O = Point('O')
    O.set_vel(inertial_frame, 0)

    C = Point('C')
    C.set_pos(O, x * inertial_frame.x)

    P1 = Point('P1')
    P1.set_pos(C, -l1 * mass1_frame.y)

    P2 = Point('P2')
    P2.set_pos(P1, -l2 * mass2_frame.y)

    Cart = Particle('Cart', C, m)
    Mass1 = Particle('Mass1', P1, m1)
    Mass2 = Particle('Mass2', P2, m2)

    coords = [x, theta1, theta2]
    speeds = [v, omega1, omega2]

    kinematics = [v - x.diff(),
                  omega1 - theta1.diff(),
                  omega2 - theta2.diff()]

    Fx = dynamicsymbols('Fx')
    cart_force = (C, Fx * inertial_frame.x)
    mass1_force = (P1, -m1 * g * inertial_frame.y)
    mass2_force = (P2, -m2 * g * inertial_frame.y)

    kane = KanesMethod(inertial_frame, q_ind=coords, u_ind=speeds, kd_eqs=kinematics)
    bodies = [Cart, Mass1, Mass2]
    loads = [cart_force, mass1_force, mass2_force]

    fr, frstar = kane.kanes_equations(bodies, loads)
    mass_matrix = trigsimp(kane.mass_matrix_full)
    forcing_vec = trigsimp(kane.forcing_full)
    # print(mass_matrix)
    # print(forcing_vec)

    constants = [m, m1, m2, l1, l2, g]
    specified = [Fx]

    right_hand_side = generate_ode_function(forcing_vec, coords, speeds, constants,
                                            mass_matrix=mass_matrix, specifieds=specified)

    const_val = np.array([10,
                          5,
                          5,
                          1,
                          1,
                          9.81])

    F_max = 80.
    act_time = 5.
    spec_val = lambda x, t: [F_max] if t < act_time/2 else [-F_max] if t < act_time else [0.]

    x0 = np.zeros(6)
    t_step = 0.01
    time = np.arange(0, 2*act_time + t_step, t_step)

    y = odeint(right_hand_side, x0, time, args=(spec_val, const_val))

    cart_shape = Box(color='black', width=0.4, height=0.2, depth=0.2)
    mass1_shape = Sphere(color='black', radius=0.1)
    mass2_shape = Sphere(color='black', radius=0.1)
    rod1_shape = Cylinder(color='blue', radius=0.05, length=const_val[3])
    rod2_shape = Cylinder(color='green', radius=0.05, length=const_val[4])

    R1 = Point('R1')
    R1.set_pos(C, -l1/2 * mass1_frame.y)
    R2 = Point('R2')
    R2.set_pos(P1, -l2/2 * mass2_frame.y)

    cart_viz_frame = VisualizationFrame(inertial_frame, C, cart_shape)
    mass1_viz_frame = VisualizationFrame(inertial_frame, P1, mass1_shape)
    mass2_viz_frame = VisualizationFrame(inertial_frame, P2, mass2_shape)
    rod1_viz_frame = VisualizationFrame(mass1_frame, R1, rod1_shape)
    rod2_viz_frame = VisualizationFrame(mass2_frame, R2, rod2_shape)

    scene = Scene(inertial_frame, O)
    scene.visualization_frames = [cart_viz_frame,
                                  mass1_viz_frame,
                                  mass2_viz_frame,
                                  rod1_viz_frame,
                                  rod2_viz_frame]

    scene.states_symbols = coords + speeds
    scene.constants = dict(zip(constants, const_val))
    scene.states_trajectories = y
    scene.display()

    P1_x = P1.pos_from(O).dot(inertial_frame.x)
    P2_x = P2.pos_from(O).dot(inertial_frame.x)
    pos1_func = lambdify((x, theta1, l1), P1_x, 'numpy')
    pos2_func = lambdify((x, theta1, theta2, l1, l2), P2_x, 'numpy')

    plt.plot(time, y[:, 0])
    plt.plot(time, pos1_func(y[:,0], y[:,1], const_val[3]))
    plt.plot(time, pos2_func(y[:, 0], y[:, 1], y[:,2], const_val[3], const_val[4]))
    plt.xlabel('Time [s]')
    plt.ylabel('X [deg]')
    plt.legend(['Cart', 'Mass1', 'Mass2'])
    plt.show()
