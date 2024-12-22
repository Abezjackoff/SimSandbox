import matplotlib.pyplot as plt
import numpy as np
from pydy.codegen.ode_function_generators import generate_ode_function
from sympy import symbols, simplify, trigsimp, Matrix, matrix2numpy, latex, lambdify, nsimplify
from sympy.physics.mechanics import Particle, KanesMethod
from sympy.physics.vector import dynamicsymbols, ReferenceFrame, Point, dot, cross, init_vprinting
from scipy.integrate import odeint
from scipy.linalg import inv, solve_continuous_are
from pydy.viz.shapes import Cylinder, Sphere, Box
from pydy.viz.visualization_frame import VisualizationFrame
from pydy.viz.scene import Scene


class PlantMechanicsModel:
    frames = dict()
    points = dict()
    coords = []
    speeds = []
    constants = []
    specified = []
    kane = None
    const_dict = dict()

    def add_frame(self, name):
        if (name not in self.frames) or (not self.frames[name]):
            F = ReferenceFrame('name')
            self.frames[name] = F
            return F
        else:
            raise Exception('Overwriting frames is not allowed!')

    def add_point(self, name: str):
        if (name not in self.points) or (not self.points[name]):
            P = Point(name)
            self.points[name] = P
            return P
        else:
            raise Exception('Overwriting points is not allowed!')

    def build_system_dynamics(self):

        x, theta1, theta2 = dynamicsymbols('x theta1 theta2')
        v, omega1, omega2 = dynamicsymbols('v omega1 omega2')
        self.coords = [x, theta1, theta2]
        self.speeds = [v, omega1, omega2]

        m, m1, m2 = symbols('m m1 m2')
        g, l1, l2 = symbols('g l1 l2')
        self.constants = [m, m1, m2, l1, l2, g]

        Fx = dynamicsymbols('Fx')
        self.specified = [Fx]

        self.add_frame('I')
        self.add_frame('P1')
        self.frames['P1'].orient(self.frames['I'], 'Axis', (theta1, self.frames['I'].z))
        self.add_frame('P2')
        self.frames['P2'].orient(self.frames['I'], 'Axis', (theta2, self.frames['I'].z))

        self.add_point('O')
        self.points['O'].set_vel(self.frames['I'], 0)
        self.add_point('C')
        self.points['C'].set_pos(self.points['O'], x * self.frames['I'].x)
        self.add_point('P1')
        self.points['P1'].set_pos(self.points['C'], -l1 * self.frames['P1'].y)
        self.add_point('P2')
        self.points['P2'].set_pos(self.points['P1'], -l2 * self.frames['P2'].y)

        Cart = Particle('Cart', self.points['C'], m)
        Mass1 = Particle('Mass1', self.points['P1'], m1)
        Mass2 = Particle('Mass2', self.points['P2'], m2)

        kinematics = [v - x.diff(),
                      omega1 - theta1.diff(),
                      omega2 - theta2.diff()]


        cart_force = (self.points['C'], Fx * self.frames['I'].x)
        mass1_force = (self.points['P1'], -m1 * g * self.frames['I'].y)
        mass2_force = (self.points['P2'], -m2 * g * self.frames['I'].y)

        self.kane = KanesMethod(self.frames['I'], q_ind=self.coords, u_ind=self.speeds, kd_eqs=kinematics)
        bodies = [Cart, Mass1, Mass2]
        loads = [cart_force, mass1_force, mass2_force]

        fr, frstar = self.kane.kanes_equations(bodies, loads)
        mass_matrix = trigsimp(self.kane.mass_matrix_full)
        forcing_vec = trigsimp(self.kane.forcing_full)
        # print(mass_matrix)
        # print(forcing_vec)

        right_hand_side = generate_ode_function(forcing_vec, self.coords, self.speeds, self.constants,
                                                mass_matrix=mass_matrix, specifieds=self.specified)

        return right_hand_side

    def set_system_parameters(self, const_vals):
        self.const_dict = dict(zip(self.constants, const_vals))

    def get_lqr_gains(self, equilibrium_point, Q, R):
        equilibrium_dict = dict(zip(self.coords + self.speeds, equilibrium_point))

        linearizer = self.kane.to_linearizer()
        linearizer.r = Matrix(self.specified)
        A, B = linearizer.linearize(op_point=[equilibrium_dict, self.const_dict], A_and_B=True)
        A = nsimplify(A, tolerance=1e-12, rational=True)
        A = matrix2numpy(A, dtype='float')
        B = matrix2numpy(B, dtype='float')

        S = solve_continuous_are(A, B, Q, R)
        K = np.dot(np.dot(inv(R), B.T), S)

        return K


def pydy_viz(plant, y):
    cart_shape = Box(color='black', width=0.4, height=0.2, depth=0.2)
    mass1_shape = Sphere(color='black', radius=0.1)
    mass2_shape = Sphere(color='black', radius=0.1)
    rod1_shape = Cylinder(color='blue', radius=0.05, length=plant.constants[3])
    rod2_shape = Cylinder(color='green', radius=0.05, length=plant.constants[4])

    R1 = Point('R1')
    R1.set_pos(plant.points['C'], -plant.constants[3]/2 * plant.frames['P1'].y)
    R2 = Point('R2')
    R2.set_pos(plant.points['P1'], -plant.constants[4]/2 * plant.frames['P2'].y)

    cart_viz_frame = VisualizationFrame(plant.frames['I'], plant.points['C'], cart_shape)
    mass1_viz_frame = VisualizationFrame(plant.frames['I'], plant.points['P1'], mass1_shape)
    mass2_viz_frame = VisualizationFrame(plant.frames['I'], plant.points['P2'], mass2_shape)
    rod1_viz_frame = VisualizationFrame(plant.frames['P1'], R1, rod1_shape)
    rod2_viz_frame = VisualizationFrame(plant.frames['P2'], R2, rod2_shape)

    scene = Scene(plant.frames['I'], plant.points['O'])
    scene.visualization_frames = [cart_viz_frame,
                                  mass1_viz_frame,
                                  mass2_viz_frame,
                                  rod1_viz_frame,
                                  rod2_viz_frame]

    scene.states_symbols = plant.coords + plant.speeds
    scene.constants = plant.const_dict
    scene.states_trajectories = y
    scene.display()


if __name__ == '__main__':

    plant = PlantMechanicsModel()
    rhs = plant.build_system_dynamics()
    # help(rhs)

    plant.set_system_parameters(np.array([10, 5, 5, 1, 1, 9.81]))

    F_max = 80.
    act_time = 5.
    spec_ctrl = lambda x, t: [F_max] if t < act_time/2 else [-F_max] if t < act_time else [0.]

    x0 = np.zeros(6)
    x0[0] = 2
    x0[1] = np.pi - np.deg2rad(10)
    x0[2] = np.pi + np.deg2rad(10)
    t_step = 0.01
    time = np.arange(0, 2*act_time + t_step, t_step)

    x_targ = np.zeros(6)
    x_targ[0] = 0
    x_targ[1] = np.pi
    x_targ[2] = np.pi

    Q = np.diag([5, 20, 20, 1, 10, 10])
    R = 0.01 * np.eye(1)
    K = plant.get_lqr_gains(x_targ, Q, R)
    lqr_ctrl = lambda x, t: np.dot(K, (x_targ - x).T)

    y = odeint(rhs, x0, time, args=(lqr_ctrl, plant.const_dict))

    pydy_viz(plant, y)

    P1_x = plant.points['P1'].pos_from(plant.points['O']).dot(plant.frames['I'].x)
    P2_x = plant.points['P2'].pos_from(plant.points['O']).dot(plant.frames['I'].x)
    pos1_func = lambdify((plant.coords[0], plant.coords[1], plant.constants[3]), P1_x, 'numpy')
    pos2_func = lambdify((plant.coords[0], plant.coords[1], plant.coords[2],
                          plant.constants[3], plant.constants[4]), P2_x, 'numpy')

    l1, l2 = symbols('l1 l2')
    plt.plot(time, y[:, 0])
    plt.plot(time, pos1_func(y[:,0], y[:,1], plant.const_dict[plant.constants[3]]))
    plt.plot(time, pos2_func(y[:, 0], y[:, 1], y[:,2],
                             plant.const_dict[plant.constants[3]], plant.const_dict[plant.constants[4]]))
    plt.xlabel('Time [s]')
    plt.ylabel('X [deg]')
    plt.legend(['Cart', 'Mass1', 'Mass2'])
    plt.show()
