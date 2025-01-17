import abc
import numpy as np
from scipy.integrate import odeint
from scipy import optimize

from dynamics.mech_sys_plant import PlantMechanicsModel


class MPController:
    """MPController base class for managing model predictive control.

    It provides common methods to set up and run MPC on a given plant.

    Parameters
    ----------
    plant : PlantMechanicsModel
        The physical system being controlled.
    y0 : array_like
        Initial condition vector for the system.
    dT : float
        Sample time between consecutive control updates.
    finT : float
        Final simulation time duration.
    predT : float
        Prediction horizon length for future state estimation.
    ctrlT : float
        Control horizon length for determining optimal control actions within each sample.
    stride : int, optional
        Number of control samples applied and stepped over after each iteration of MPC. Default is 1.

    Attributes
    ----------
    plant : PlantMechanicsModel
        The physical system being controlled.
    y0 : array_like
        Initial condition vector for the system.
    dT : float
        Sample time between consecutive control updates.
    finT : float
        Final simulation time duration.
    time : numpy.ndarray
        Array representing the discretized time samples for prediction horizon.
    n_pred : int
        Number of samples in the prediction horizon.
    n_ctrl : int
        Number of samples in the control horizon.
    n_step : int
        Number of control samples applied and stepped over after each iteration of MPC.
    u : numpy.ndarray
        Control input sequence matrix where rows represent different control samples.
    u_min : array_like
        Minimum allowed values for control inputs. If not provided, no lower bound constraints are enforced.
    u_max : array_like
        Maximum allowed values for control inputs. If not provided, no upper bound constraints are enforced.
    has_bounds : bool
        Flag indicating whether there are any bounds specified for control inputs.
    cost : float
        Current cost value computed during MPC iterations.
    iter : int
        Counter tracking the number of times MPC results have been displayed when the callback is triggered.

    Examples
    --------
    mpc = MPController(plant, y0, dT, finT, predT, ctrlT) |
    mpc.set_u_bounds([-u_max], [u_max]) |
    mpc.init_u() |
    t, y, u = mpc.run() |
    """

    def __init__(self, plant: PlantMechanicsModel, y0, dT, finT, predT, ctrlT, stride=1):
        self.plant = plant
        self.y0 = y0
        self.dT = dT
        self.finT = finT
        self.time = np.arange(0, predT + dT, dT)
        self.n_pred = self.time.shape[0]
        self.n_ctrl = min(int(np.round(ctrlT / dT)), self.n_pred - 1)
        self.n_step = min(max(stride, 1), self.n_ctrl)
        self.u = np.zeros((self.n_ctrl, self.plant.n_u))
        self.u_min = None
        self.u_max = None
        self.has_bounds = False
        self.cost = 0
        self.iter = 0

    def set_u_bounds(self, u_min, u_max):
        self.u_min = (u_min * np.ones((self.n_ctrl, 1))).flatten()
        self.u_max = (u_max * np.ones((self.n_ctrl, 1))).flatten()
        self.has_bounds = True

    def get_u_lut(self, x, t):
        i = int(np.floor(t / self.dT))
        i = min(i, self.n_ctrl - 1)
        return self.u[i]

    def set_u_lut(self, func_u, y=None):
        for i in range(self.n_ctrl):
            if y is None:
                x = 0
            else:
                x = y[i]
            t = self.time[i]
            self.u[i] = func_u(x, t)

    def get_sensitivity(self, y, u):
        """Compute the sensitivity matrix G.

        The sensitivity is computed based on provided state trajectory and control inputs
        using the linearization of the plant dynamics along this trajectory.

        Parameters
        ----------
        y : numpy.ndarray
            The system trajectory over the prediction horizon.
            Each row of y represents a state vector at a particular time point.

        u : numpy.ndarray
            The control inputs over the control horizon.
            Each row of u represents a control input vector at a particular time point.

        Returns
        -------
        numpy.ndarray
            A 4-dimensional array representing the sensitivity matrix G, with dimensions
            `(self.n_x, self.n_u, self.n_pred, self.n_ctrl)`. Each 2-d submatrix `G[k, l][:, :]`
            represents the partial derivatives of the k-th state variable
            with respect to the l-th control input at different relative points in time
            of prediction and control horizons.
        """
        G = np.zeros((self.plant.n_x, self.plant.n_u, self.n_pred, self.n_ctrl))

        A_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_x))
        B_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_u))
        AA_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_x))
        AB_trj = np.zeros((self.n_pred-1, self.plant.n_x, self.plant.n_u))
        for i in range(0, self.n_pred-1):
            x = np.array(y[i])
            r = np.array(u[min(i, self.n_ctrl - 1)])
            A, B = self.plant.get_lde_matrices(x, r)
            A_trj[i] = A
            B_trj[i] = B
            # x = np.array(y[i])
            # A, B = plant.get_lde_matrices(x, r)
            # A_trj[i] += A
            # B_trj[i] += B
            # A_trj[i] *= 0.5
            # B_trj[i] *= 0.5
            AA_trj[i] = A_trj[i] @ A_trj[i]
            AB_trj[i] = A_trj[i] @ B_trj[i]

        for j in range(0, self.n_ctrl):
            for i in range(j+1, self.n_pred):
                if i == j+1:
                    z = B_trj[i-1] * self.dT + 0.5 * AB_trj[i-1] * self.dT ** 2
                else:
                    z = np.zeros((self.plant.n_x, self.plant.n_u))
                    for k in range(0, self.plant.n_x):
                        for l in range(0, self.plant.n_u):
                            z[k, l] = G[k, l][i-1, j]
                    z = (np.eye(self.plant.n_x) + A_trj[i-1] * self.dT + 0.5 * AA_trj[i-1] * self.dT**2) @ z

                for k in range(0, self.plant.n_x):
                    for l in range(0, self.plant.n_u):
                        G[k, l][i, j] = z[k, l]
        return G

    @abc.abstractmethod
    def calc_cost(self, y):
        """An abstract method to compute cost.

        It is supposed to return the cost function value
        depending on the state trajectory y and control inputs `self.u`
        This method must be implemented by subclasses.

        Parameters
        ----------
        y : numpy.ndarray
            The state trajectory for cost computation.
        """
        pass

    @abc.abstractmethod
    def calc_grad(self, y):
        """An abstract method to compute gradient of cost function.

        It is supposed to return the gradient of the cost with respect to control inputs u
        This method must be implemented by subclasses.

        Parameters
        ----------
        y : numpy.ndarray
            The state trajectory for gradient computation.
        """
        pass

    @abc.abstractmethod
    def calc_hess(self, y):
        """An abstract method to compute hessian of cost function.

        It is supposed to return the hessian of the cost with respect to control inputs u
        This method must be implemented by subclasses.

        Parameters
        ----------
        y : numpy.ndarray
            The state trajectory for hessian computation.
        """
        pass

    def get_J_and_DJ(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        J = self.calc_cost(y)
        DJ = self.calc_grad(y)
        self.cost = J
        return (J, DJ)

    def get_J_func(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        J = self.calc_cost(y)
        self.cost = J
        return J

    def get_J_grad(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        DJ = self.calc_grad(y)
        return DJ

    def get_J_hess(self, u):
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        y = odeint(self.plant.rhs, self.y0, self.time, args=(self.get_u_lut, self.plant.const_dict))
        DDJ = self.calc_hess(y)
        return DDJ

    def print_progress(self, x, e=None, context=None):
        self.iter += 1
        print(f'#{self.iter}   {self.cost}')

    def init_u(self):
        """Placeholder method for initializing the control input `self.u`.

        This method initializes the control input sequence based on some trajectory
        assumptions or heuristic approach. It currently contains commented-out code
        that suggests options for initializing `self.u`.
        """
        # u = self.u.flatten()

        # J, DJ = self.J_and_DJ(u)
        # print(J)
        # for i in range(2):
        #     DDJ = self.J_hess(u)
        #     print(np.linalg.cond(DDJ))
        #     u = solve(DDJ, -DJ)
        #     J, DJ = self.J_and_DJ(u)
        #     print(J)

        # if self.has_bounds:
        #     bounds = optimize.Bounds(self.u_min, self.u_max)
        # else:
        #     bounds = None
        # res = optimize.direct(self.J_func, bounds, locally_biased=False, maxiter=100, callback=self.print_progress)
        # u = res.x
        # print(res.fun)

        # self.u = u.reshape((self.n_ctrl, self.plant.n_u))
        pass

    def optimize_u(self):
        """Placeholder method for optimizing the control sequence `self.u`.

        This method solves the optimization problem defined by the cost function and its derivatives.
        Its implementation could employ various algorithms such as BFGS, Newton-CG,
        or other techniques including constrained optimization like L-BFGS-B or TNC.
        The method curently contains code that suggests 'trust-constr' solver as the most versatile algorithm
        but in general it is supposed to be adjusted for a specific MPC subclass.
        """

        u = self.u.flatten()

        if self.has_bounds:
            bounds = optimize.Bounds(self.u_min, self.u_max)
        else:
            bounds = None
        res = optimize.minimize(self.get_J_and_DJ, u, method='trust-constr', jac=True, hess=self.get_J_hess, bounds=bounds,
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

    def run(self):
        t_full = np.array([])
        y_full = np.zeros((0, self.plant.n_x))
        u_full = np.zeros((0, self.plant.n_u))

        self.optimize_u()
        y = odeint(self.plant.rhs, self.y0, self.time[0:self.n_step+1], args=(self.get_u_lut, self.plant.const_dict))
        t_full = np.append(t_full, self.time[0:self.n_step])
        y_full = np.append(y_full, y[0:self.n_step, :], axis=0)
        u_full = np.append(u_full, self.u[0:self.n_step, :], axis=0)

        step = self.n_step * self.dT
        t_next = step
        while t_next < self.finT:
            self.y0 = y[-1]
            self.u = np.roll(self.u, -self.n_step, axis=0)
            self.u[-self.n_step:, :] = self.u[-self.n_step-1, :]

            self.optimize_u()
            y = odeint(self.plant.rhs, self.y0, self.time[0:self.n_step+1], args=(self.get_u_lut, self.plant.const_dict))
            t_full = np.append(t_full, t_full[-self.n_step:] + step)
            y_full = np.append(y_full, y[0:self.n_step, :], axis=0)
            u_full = np.append(u_full, self.u[0:self.n_step, :], axis=0)

            t_next += step

        t_full = np.append(t_full, t_full[-1] + self.dT)
        y_full = np.append(y_full, y[-1:, :], axis=0)
        u_full = np.append(u_full, u_full[-1:, :], axis=0)
        return t_full, y_full, u_full
