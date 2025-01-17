import abc
import numpy as np
from sympy import Matrix, matrix2numpy, nsimplify
from sympy.physics.vector import ReferenceFrame, Point
from scipy.linalg import inv, solve_continuous_are

class PlantMechanicsModel:
    """Base class for controllable mechanical system.

    Attributes
    ----------
    frames : dict[str, ReferenceFrame]
        Dictionary of reference frames used to define the system dynamics.
    points : dict[str, Point]
        Dictionary of points used to define reference points of the system.
    const_dict : dict[Symbol, float]
        Dictionary of constant values that are used to evaluate expressions.
    coords : list[Symbol]
        List of generalized coordinates.
    speeds : list[Symbol]
        List of generalized speeds.
    constants : list[Symbol]
        List of constant parameters.
    specified : list[Symbol]
        List of specified forces.
    kane : KanesMethod
        Kane object from `sympy.physics.mechanics`.
    rhs : Function
        Right-hand side function of the system dynamics.
    n_x : int
        Number of states equal to `len(self.coords) + len(self.speeds)`.
    n_u : int
        Number of inputs equal to `len(self.specified)`.

    Examples
    --------
    plant = PlantMechanicsModel() |
    rhs = plant.build_system_dynamics() |
    plant.set_system_parameters(params_array) |
    """

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
        """Abstract method to build the system dynamics.

        The method defines mechanical parts of the system and constraints between them.
        It builds the right-hand side function of the system dynamics and stores it into `self.rhs` attribute.
        """
        pass

    @abc.abstractmethod
    def get_lde_matrices(self, x, u):
        """Abstract method to get linear differential equation (LDE) matrices A and B.

        A and B describe the linearized system dynamics around an operating point so that dx/dt = Ax + Bu.
        This method should be implemented by subclasses to define how the matrices are obtained.

        Parameters
        ----------
        x : numpy.ndarray
            Vector of state variables.
        u : numpy.ndarray
            Vector of control inputs.
        """
        pass
