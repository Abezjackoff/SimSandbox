import abc
import numpy as np
from sympy import Matrix, matrix2numpy, nsimplify
from sympy.physics.vector import ReferenceFrame, Point
from scipy.linalg import inv, solve_continuous_are

class PlantMechanicsModel:

    def __init__(self):
        self.frames = dict()
        self.points = dict()
        self.const_dict = dict()
        self.coords = []
        self.speeds = []
        self.constants = []
        self.specified = []
        self.kane = None
        self.rhs = None
        self.n_x = 0
        self.n_u = 0

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

        P = solve_continuous_are(A, B, Q, R)
        K = np.dot(np.dot(inv(R), B.T), P)

        return K

    @abc.abstractmethod
    def build_system_dynamics(self):
        pass

    @abc.abstractmethod
    def get_lde_matrices(self, x, u):
        pass
