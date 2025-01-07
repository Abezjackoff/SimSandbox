import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.integrate import odeint

from dyn_sys_plant import PlantMechanicsModel
from mpc import MPController


def blur_img(img):
    top = img[:-2, 1:-1]
    left = img[1:-1, :-2]
    center = img[1:-1, 1:-1]
    bottom = img[2:, 1:-1]
    right = img[1:-1, 2:]
    return (top + left + center + bottom + right) / 5

def do_blurring(name: str):
    img = plt.imread(name)

    res = timeit.timeit(lambda: blur_img(img), number=100)
    print(res / 100)

    blurred = blur_img(img)
    for _ in range(10):
        blurred = blur_img(blurred)

    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(blurred)

    plt.show()


def get_motion_solution(u, dT, n):
    x = np.zeros(n+1)
    v = np.zeros(n+1)
    for i in range(1, n+1):
        v[i] = v[i-1] + u[i-1] * dT
        x[i] = x[i-1] + (v[i-1] + v[i]) * dT / 2
    return x, v


class PointMassPlant(PlantMechanicsModel):
    def __init__(self):
        super().__init__()
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1]])

    def build_system_dynamics(self):
        self.n_x = 2
        self.n_u = 1
        def ode_rhs(x, t, r, p=None):
            return self.A @ x + self.B @ r(x, t)

        self.rhs = ode_rhs
        return ode_rhs

    def get_lde_matrices(self, x, u):
        return self.A, self.B


class PointMassMPC(MPController):

    def __init__(self, a, plant: PlantMechanicsModel, y0, dT, finT, predT, ctrlT):
        super().__init__(plant, y0, dT, finT, predT, ctrlT)
        self.a = a

    def get_cost(self, y):
        x = y[:, 0]
        v = y[:, 1]

        J = 0.5 * np.sum((x-1)**2 + self.a * v**2)
        return J

    def get_grad(self, y):
        x = y[:, 0]
        v = y[:, 1]

        dJ_dx = x - 1
        dJ_dv = self.a * v

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gv = G[1, 0]

        DJ = dJ_dx @ Gx + dJ_dv @ Gv
        return DJ

    def get_hess(self, y):
        x = y[:, 0]
        v = y[:, 1]

        ddJ_dxdx = np.ones(x.shape)
        ddJ_dvdv = self.a * np.ones(v.shape)

        G = self.get_sensitivity(y, self.u)
        Gx = G[0, 0]
        Gv = G[1, 0]

        DDJ = Gx.T @ (ddJ_dxdx * Gx.T).T + Gv.T @ (ddJ_dvdv * Gv.T).T
        return DDJ

    def init_u(self):
        u = self.u.flatten()
        J, DJ = self.J_and_DJ(u)
        print(J)
        DDJ = self.J_hess(u)
        # print(np.linalg.cond(DDJ))
        u = solve(DDJ, -DJ)
        print(self.J_and_DJ(u)[0])
        self.u = u.reshape((self.n_ctrl, self.plant.n_u))

if __name__ == '__main__':

    # do_blurring('resources/starbucks-logo.png')

    dT = 2
    dT2 = dT*dT
    a = 5
    A = np.array([[14*dT2+a, 8*dT2+a, 3*dT2+a, a],
                  [8*dT2+a, 5*dT2+a, 2*dT2+a, a],
                  [3*dT2+a, 2*dT2+a, dT2+a, a],
                  [a, a, a, a]])
    B = np.array([[6], [3], [1], [0]])
    X = solve(A, B, assume_a='sym')
    print(X)

    n_steps = 20
    u = np.zeros(n_steps)
    x0 = np.zeros(2)

    plant = PointMassPlant()
    plant.build_system_dynamics()
    mpc = PointMassMPC(a, plant, x0, dT, n_steps * dT, n_steps * dT, n_steps * dT)
    mpc.init_u()

    # res = minimize(mpc.J_and_DJ, mpc.u, method='Newton-CG', jac=True, hess=mpc.J_hess, options={'disp': True})
    # res = minimize(mpc.J_and_DJ, mpc.u, method='BFGS', jac=True, options={'disp': True})
    # res = minimize(J_and_DJ, np.zeros(n_steps), method='BFGS', jac=True, options={'disp': True})
    # res = minimize(J_func, np.zeros(n_steps), method='nelder-mead', options={'disp': True})
    # print(res.x)

    # x, v = get_motion_solution(mpc.u, dT, n_steps)
    y = odeint(plant.rhs, x0, mpc.time, args=(mpc.get_u_lut,))
    plt.figure('Response')
    plt.plot(np.linspace(0, n_steps, n_steps+1), y[:, 0])
    plt.plot(np.linspace(0, n_steps, n_steps+1), y[:, 1])
    plt.figure('Control')
    plt.plot(mpc.u[:, 0])
    plt.show()
