import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from matplotlib import animation, colors, colormaps


DT = 1e-19
SIM_LEN = 5000
EPS = 1e-12
COULOMB_K = 2.533e38

SIM_SPEED = 8

E_PLOT_N = 100


def coulomb_law(x1, x2, y1, y2, q1, q2, m1):
    # acceleration of particle 1
    s = COULOMB_K * q1 * q2 / (m1 * np.linalg.norm([x1 - x2, y1 - y2])**3 + EPS)
    return s * np.array([x1 - x2, y1 - y2])

def get_derivative(state):
    d = np.zeros_like(state)
    n = state.shape[0]
    for i in range(n):
        d[i, 0:2] = state[i, 2:4]
        xi = state[i, 0]
        yi = state[i, 1]
        for j in range(n):
            if i == j:
                continue
            xj = state[j, 0]
            yj = state[j, 1]
            d[i, 2:4] += coulomb_law(xi, xj, yi, yj, q[i], q[j], m[i])
    return d

def simulate_steps(state0, h, steps):
    simulation = [state0]

    # state = state0
    # for _ in range(steps):
    #     state += h * get_derivative(state)
    #     simulation.append(np.copy(state))

    y0 = state0.flatten()
    solver = RK45(lambda t,y: get_derivative(np.reshape(y, state0.shape)).flatten(),
                  0, y0, t_bound = h * (steps + 1), max_step = h)

    for _ in range(steps):
        solver.step()
        state = np.reshape(solver.y, state0.shape)
        simulation.append(np.copy(state))

    return np.array(simulation)

def E_field(state, q, X, Y):
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    for i in range(state.shape[0]):
        xi = state[i, 0]
        yi = state[i, 1]
        r2 = np.square(X - xi) + np.square(Y - yi)
        Ex += COULOMB_K * q[i] * (X - xi) / (r2 ** 3/2 + EPS)
        Ey += COULOMB_K * q[i] * (Y - yi) / (r2 ** 3/2 + EPS)
    return Ex, Ey

def animate_func(i):
    Ex, Ey = E_field(simulation[i * SIM_SPEED], q, X, Y)
    E_strength = np.log(Ex * Ex + Ey * Ey + EPS)
    mesh.set_array(E_strength)
    scatter.set_offsets(simulation[i * SIM_SPEED][:, 0:2])
    return scatter, mesh


if __name__ == '__main__':
    state = np.zeros((3, 4))
    m = np.array([938, 0.511, 938])
    q = np.array([1, -1, 1])

    dist = 52.9
    state[1, 0] = dist * np.cos(1.1)
    state[1, 1] = dist * np.sin(1.1)
    acc = coulomb_law(state[1, 0], state[0, 0], state[1, 1], state[0, 1], q[1], q[0], m[1])
    v0 = np.sqrt(np.linalg.norm(acc) * dist)
    state[1, 2] = v0 * np.sin(1.1)
    state[1, 3] = -v0 * np.cos(1.1)
    state[2, 0] = -190
    state[2, 2] = v0 * 0.4
    state[:, 0] -= 20

    bound = 200
    offset = -20

    x = np.linspace(-bound + offset, bound + offset, E_PLOT_N)
    y = np.linspace(-bound, bound, E_PLOT_N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    Ex, Ey = E_field(state, q, X, Y)
    E_strength = np.log(Ex * Ex + Ey * Ey + EPS)
    mesh = plt.pcolormesh(X, Y, E_strength, cmap = 'inferno')

    scatter = plt.scatter(state[:, 0], state[:, 1], s=np.log(m / np.min(m) + 1) * 15,
                          c=q, cmap='seismic', vmin=-2, vmax=2)
    axs = fig.get_axes()
    axs[0].set_xlim(-bound + offset, bound + offset)
    axs[0].set_ylim(-bound, bound)
    plt.gca().set_aspect('equal')

    simulation = simulate_steps(state, DT, SIM_LEN)

    anim = animation.FuncAnimation(fig, animate_func, frames = range(SIM_LEN // SIM_SPEED), interval = 40)

    fig.set_size_inches(6, 6)
    fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = None, hspace = None)
    plt.axis('off')
    plt.show()
