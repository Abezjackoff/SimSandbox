import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, simplify, trigsimp, Matrix, matrix2numpy, latex
from sympy.abc import c, d, e, f, g, h, theta
from sympy.physics.mechanics import inertia, RigidBody, KanesMethod
from sympy.physics.vector import dynamicsymbols, ReferenceFrame, Point, dot, cross, init_vprinting
from pydy.codegen.ode_function_generators import generate_ode_function
from pydy.viz.shapes import Cylinder, Sphere
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene
from scipy.integrate import odeint
from scipy.linalg import inv, solve_continuous_are

def blur_img(img):
    top = img[:-2, 1:-1]
    left = img[1:-1, :-2]
    center = img[1:-1, 1:-1]
    bottom = img[2:, 1:-1]
    right = img[1:-1, 2:]
    return (top + left + center + bottom + right) / 5

if __name__ == '__main__':

    # x = np.arange(-np.pi, np.pi, 0.1)
    # y1 = 2 * np.sin(x)
    # y2 = x + np.cos(4*x)
    # plt.figure()
    # plt.plot(x, y1, 'r')
    # plt.plot(x, y2, '--b')
    # plt.show()

    # A = ReferenceFrame('N')
    # B = A.orientnew('B', 'Axis', (theta, A.z))
    # a = c * A.x + d * A.y + e * A.z
    # b = f * B.x + g * B.y + h * B.z
    # print(dot(a, b))
    # print(cross(a, b))
    # print(b.express(A))
    #
    # Jx, Jy, Jz = symbols('Jx Jy Jz')
    # J = inertia(A, Jx, Jy, Jz, 0, 0, 0)
    # print(J.express(B))

    inertial_frame = ReferenceFrame('I')
    lower_leg_frame = ReferenceFrame('L')
    upper_leg_frame = ReferenceFrame('U')
    torso_frame = ReferenceFrame('T')

    theta1, theta2, theta3 = dynamicsymbols('theta1 theta2 theta3')

    lower_leg_frame.orient(inertial_frame, 'Axis', (theta1, inertial_frame.z))
    upper_leg_frame.orient(lower_leg_frame, 'Axis', (theta2, lower_leg_frame.z))
    torso_frame.orient(upper_leg_frame, 'Axis', (theta3, upper_leg_frame.z))
    print(lower_leg_frame.dcm(inertial_frame))
    print(upper_leg_frame.dcm(inertial_frame).simplify())

    ankle = Point('A')
    lower_leg_length = symbols('l_L')

    knee = Point('K')
    knee.set_pos(ankle, lower_leg_length * lower_leg_frame.y)
    print(knee.pos_from(ankle).express(inertial_frame).simplify())
    upper_leg_length = symbols('l_U')

    hip = Point('H')
    hip.set_pos(knee, upper_leg_length * upper_leg_frame.y)

    lower_leg_com_length, upper_leg_com_length, torso_com_length = symbols('d_L, d_U, d_T')
    lower_leg_com = Point('L_o')
    lower_leg_com.set_pos(ankle, lower_leg_com_length * lower_leg_frame.y)
    upper_leg_com = Point('U_o')
    upper_leg_com.set_pos(knee, upper_leg_com_length * upper_leg_frame.y)
    torso_com = Point('T_o')
    torso_com.set_pos(hip, torso_com_length * torso_frame.y)

    omega1, omega2, omega3 = dynamicsymbols('omega1 omega2 omega3')
    kinematical_differential_equations = [  omega1 - theta1.diff(),
                                            omega2 - theta2.diff(),
                                            omega3 - theta3.diff()  ]

    lower_leg_frame.set_ang_vel(inertial_frame, omega1 * inertial_frame.z)
    upper_leg_frame.set_ang_vel(lower_leg_frame, omega2 * lower_leg_frame.z)
    torso_frame.set_ang_vel(upper_leg_frame, omega3 * upper_leg_frame.z)
    print(torso_frame.ang_vel_in(inertial_frame).express(inertial_frame))

    ankle.set_vel(inertial_frame, 0)
    lower_leg_com.v2pt_theory(ankle, inertial_frame, lower_leg_frame)
    knee.v2pt_theory(knee, inertial_frame, lower_leg_frame)
    upper_leg_com.v2pt_theory(knee, inertial_frame, upper_leg_frame)
    hip.v2pt_theory(knee, inertial_frame, upper_leg_frame)
    torso_com.v2pt_theory(hip, inertial_frame, torso_frame)
    print(torso_com.vel(inertial_frame))

    lower_leg_mass, upper_leg_mass, torso_mass = symbols('m_L m_U m_T')
    lower_leg_inertia, upper_leg_inertia, torso_inertia = symbols('I_Lz I_Uz I_Tz')

    lower_leg_inertia_ = inertia(lower_leg_frame, 0, 0, lower_leg_inertia)
    lower_leg_inertia_central = (lower_leg_inertia_, lower_leg_com)
    upper_leg_inertia_ = inertia(upper_leg_frame, 0, 0, upper_leg_inertia)
    upper_leg_inertia_central = (upper_leg_inertia_, upper_leg_com)
    torso_inertia_ = inertia(torso_frame, 0, 0, torso_inertia)
    torso_inertia_central = (torso_inertia_, torso_com)

    lower_leg = RigidBody('Lower Leg', lower_leg_com, lower_leg_frame, lower_leg_mass, lower_leg_inertia_central)
    upper_leg = RigidBody('Upper Leg', upper_leg_com, upper_leg_frame, upper_leg_mass, upper_leg_inertia_central)
    torso = RigidBody('Torso', torso_com, torso_frame, torso_mass, torso_inertia_central)

    g = symbols('g')
    lower_leg_force = (lower_leg_com, -lower_leg_mass * g * inertial_frame.y)
    upper_leg_force = (upper_leg_com, -upper_leg_mass * g * inertial_frame.y)
    torso_force = (torso_com, -torso_mass * g * inertial_frame.y)

    ankle_torque, knee_torque, hip_torque = dynamicsymbols('T_A T_K T_H')
    lower_leg_torque = (lower_leg_frame, (ankle_torque - knee_torque) * inertial_frame.z)
    upper_leg_torque = (upper_leg_frame, (knee_torque - hip_torque) * inertial_frame.z)
    torso_torque = (torso_frame, hip_torque * inertial_frame.z)

    coordinates = [theta1, theta2, theta3]
    speeds = [omega1, omega2, omega3]

    kane = KanesMethod(inertial_frame, coordinates, speeds, kinematical_differential_equations)
    bodies = [lower_leg, upper_leg, torso]
    loads = [   lower_leg_force,
                upper_leg_force,
                torso_force,
                lower_leg_torque,
                upper_leg_torque,
                torso_torque    ]

    fr, frstar = kane.kanes_equations(bodies, loads)
    mass_matrix = trigsimp(kane.mass_matrix_full)
    forcing_vec = trigsimp(kane.forcing_full)
    print(mass_matrix.free_symbols)
    print(forcing_vec.free_symbols)

    constants = [lower_leg_length,
                 lower_leg_com_length,
                 lower_leg_mass,
                 lower_leg_inertia,
                 upper_leg_length,
                 upper_leg_com_length,
                 upper_leg_mass,
                 upper_leg_inertia,
                 torso_com_length,
                 torso_mass,
                 torso_inertia,
                 g]

    specified = [ankle_torque, knee_torque, hip_torque]

    right_hand_side = generate_ode_function(forcing_vec, coordinates, speeds, constants,
                                            mass_matrix=mass_matrix, specifieds=specified)
    # help(right_hand_side)

    x0 = np.zeros(6)
    x0[0] = -np.deg2rad(10.)
    x0[1] = np.deg2rad(20.)
    x0[2] = -np.deg2rad(20.)

    numerical_constants = np.array([0.611,
                                   0.387,
                                   6.769,
                                   0.101,
                                   0.424,
                                   0.193,
                                   17.01,
                                   0.282,
                                   0.305,
                                   32.44,
                                   1.485,
                                   9.81])

    numerical_specified = np.zeros(3)

    time = np.linspace(0, 10, 600)
    # y = odeint(right_hand_side, x0, time, args=(numerical_specified, numerical_constants))

    # plt.plot(time, np.rad2deg(y[:, :3]))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Angle [deg]')
    # plt.legend([f'${latex(c)}$' for c in coordinates])
    # plt.show()

    ankle_shape = Sphere(color='black', radius=0.1)
    knee_shape = Sphere(color='black', radius=0.1)
    hip_shape = Sphere(color='black', radius=0.1)
    head_shape = Sphere(color='black', radius=0.125)

    head = Point('N')
    head.set_pos(hip, 2 * torso_com_length * torso_frame.y)
    ankle_viz_frame = VisualizationFrame(inertial_frame, ankle, ankle_shape)
    knee_viz_frame = VisualizationFrame(inertial_frame, knee, knee_shape)
    hip_viz_frame = VisualizationFrame(inertial_frame, hip, hip_shape)
    head_viz_frame = VisualizationFrame(inertial_frame, head, head_shape)

    lower_leg_center = Point('L_c')
    lower_leg_center.set_pos(ankle, lower_leg_length / 2 * lower_leg_frame.y)
    upper_leg_center = Point('U_c')
    upper_leg_center.set_pos(knee, upper_leg_length / 2 * upper_leg_frame.y)
    torso_center = Point('T_c')
    torso_center.set_pos(hip, torso_com_length  * torso_frame.y)

    constants_dict = dict(zip(constants, numerical_constants))
    lower_leg_shape = Cylinder(color='blue', radius=0.08, length=constants_dict[lower_leg_length])
    lower_leg_viz_frame = VisualizationFrame('Lower Leg', lower_leg_frame, lower_leg_center, lower_leg_shape)
    upper_leg_shape = Cylinder(color='green', radius=0.08, length=constants_dict[upper_leg_length])
    upper_leg_viz_frame = VisualizationFrame('Upper Leg', upper_leg_frame, upper_leg_center, upper_leg_shape)
    torso_shape = Cylinder(color='red', radius=0.08, length=2 * constants_dict[torso_com_length])
    torso_viz_frame = VisualizationFrame('Torso', torso_frame, torso_center, torso_shape)

    scene = Scene(inertial_frame, ankle)
    scene.visualization_frames = [ankle_viz_frame,
                                  knee_viz_frame,
                                  hip_viz_frame,
                                  head_viz_frame,
                                  lower_leg_viz_frame,
                                  upper_leg_viz_frame,
                                  torso_viz_frame]
    scene.states_symbols = coordinates + speeds
    scene.constants = constants_dict
    # scene.states_trajectories = y
    # scene.display()

    equilibrium_point = np.array(np.zeros(6))
    equilibrium_dict = dict(zip(coordinates + speeds, equilibrium_point))

    linearizer = kane.to_linearizer()
    linearizer.r = Matrix(specified)
    A, B = linearizer.linearize(op_point=[equilibrium_dict, constants_dict], A_and_B=True)
    A = matrix2numpy(A, dtype='float')
    B = matrix2numpy(B, dtype='float')

    Q = np.eye(6)
    R = np.eye(3)
    S = solve_continuous_are(A, B, Q, R)
    K = np.dot(np.dot(inv(R), B.T), S)

    def controller(x, t):
        return -np.dot(K, x.T)

    y = odeint(right_hand_side, x0, time, args=(controller, numerical_constants))

    # plt.plot(time, np.rad2deg(y[:, :3]))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Angle [deg]')
    # plt.legend([f'${latex(c)}$' for c in coordinates])
    # plt.show()

    scene.states_trajectories = y
    scene.display()

    pass
