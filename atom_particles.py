import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from matplotlib import animation, colors, colormaps


DT = 1e-19
SIM_LEN = 5000
EPS = 1e-12
COULOMB_K = 2.533e38

SIM_SPEED = 8


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

def animate_func(i):
    scatter.set_offsets(simulation[i * SIM_SPEED])
    return scatter


if __name__ == '__main__':
    state = np.zeros((3, 4))
    m = np.array([938, 0.511, 938])
    q = np.array([1, -1, 1])

    dist = 52.9
    state[1, 0] = dist * np.cos(1.1)
    state[1, 1] = dist * np.sin(1.1)
    vi = np.sqrt(np.abs(COULOMB_K * q[0] * q[1]) / (dist * m[1]))
    state[1, 2] = vi * np.sin(1.1)
    state[1, 3] = -vi * np.cos(1.1)
    state[2, 0] = -190
    state[2, 2] = vi * 0.4
    state[:, 0] -= 20

    bound = 200
    offset = -20

    fig = plt.figure()
    scatter = plt.scatter(state[:, 0], state[:, 1], s=np.log(m / np.min(m) + 1) * 15,
                          c=q, cmap='seismic', vmin=-2, vmax=2)
    axs = fig.get_axes()
    axs[0].set_xlim(-bound + offset, bound + offset)
    axs[0].set_ylim(-bound, bound)
    plt.gca().set_aspect('equal')

    simulation = simulate_steps(state, DT, SIM_LEN)

    anim = animation.FuncAnimation(fig, animate_func, frames = range(SIM_LEN // SIM_SPEED), interval = 40)
    plt.show()
