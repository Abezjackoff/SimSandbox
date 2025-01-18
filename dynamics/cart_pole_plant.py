import numpy as np
from sympy import Symbol, symbols, trigsimp
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics import Particle, KanesMethod
from pydy.codegen.ode_function_generators import generate_ode_function
from dynamics.mech_sys_plant import PlantMechanicsModel


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
        A[3, 1] = -m1 * sin2 / dnom**2 * (-u[0] * cos - (m + m1)*g/l1 * sin - m1*l1 * x[3]**2 * sin2 / 2) / l1 \
                + 1 / dnom * (u[0] * sin - (m + m1)*g/l1 * cos - m1*l1 * x[3]**2 * cos2) / l1
        A[3, 3] = -m1 * x[3] * sin2 / dnom

        B[2, 0] = 1 / dnom
        B[3, 0] = -cos / dnom / l1

        # A = np.array([[ 0.,     0.,     1.,     0.   ],
        #               [ 0.,     0.,     0.,     1.   ],
        #               [ 0.,     4.905,  0.,     0.   ],
        #               [ 0.,    14.715,  0.,     0.   ]])
        # B = np.array([[0. ],
        #               [0. ],
        #               [0.1],
        #               [0.1]])

        return A, B
