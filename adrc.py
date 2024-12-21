import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

class Plant:
    T = 10.
    A = np.matrix(f'{-1 / T}')
    B = np.matrix(f'{1 / T}')

    def get_derivative(self, x, t, u):
        drv = self.A * np.matrix(x) + self.B * np.matrix(u)
        return np.asarray(drv).ravel()

class PIDController:
    Kp = 5.
    Ki = 0.5
    i_term = 0

    def reset(self):
        self.i_term = 0

    def get_control(self, y_targ, y, t_step):
        y_err = y - y_targ
        u = -self.Kp * y_err - self.Ki * self.i_term
        self.i_term += y_err * t_step
        return u

class ADRController:
    Kp = 0.5

    b = 0.12
    w_o = 1.

    A = np.matrix('0 1; 0 0')
    B = np.matrix(f'{b}; 0')
    L = np.matrix(f'{2*w_o}; {w_o*w_o}')
    z = np.matrix(np.zeros((2, 1)))

    def eso_reset(self, y_step):
        self.z[0] = y_step
        self.z[-1] = 0

    def predict(self, u, y, t_step):
        self.z += (self.A * self.z + self.B * np.matrix(u)) * t_step
        self.z -= self.L * (self.z[0] - y) * t_step

    def get_control(self, y_targ):
        u = self.Kp * (y_targ - self.z[0].item())
        u = (u - self.z[1].item()) / self.b
        return u

if __name__ == '__main__':

    plant = Plant()
    pid = PIDController()
    adrc = ADRController()

    y_targ = 10.

    t_step = 0.1
    time = np.arange(0, 100 + t_step, t_step)

    y = [0.]
    y_step = y[0]
    pid.reset()
    control = pid.get_control(y_targ, y_step, t_step)
    u = [control]
    for t in time[:-1]:
        y_step = odeint(plant.get_derivative, y_step, [t, t + t_step],
                        args=(-5 + control,))[-1, :].item()
        y.append(y_step)
        if t == 50:
            pid.reset()
        control = pid.get_control(y_targ, y_step, t_step)
        u.append(control)

    plt.figure('Response')
    plt.plot(time, y)
    plt.xlabel('time, s')
    plt.ylabel('x')
    plt.grid()

    plt.figure('Control')
    plt.plot(time, u)
    plt.xlabel('time')
    plt.ylabel('u, s')
    plt.grid()

    y = [0.]
    y_step = y[0]
    adrc.eso_reset(y_step)
    control = adrc.get_control(y_targ)
    u = [control]
    for t in time[:-1]:
        y_step = odeint(plant.get_derivative, y_step, [t, t + t_step],
                        args=(-5 + control,))[-1, :].item()
        y.append(y_step)
        adrc.predict(control, y_step, t_step)
        if t == 50:
            adrc.eso_reset(y_step)
        control = adrc.get_control(y_targ)
        u.append(control)

    plt.figure('Response')
    plt.plot(time, y)
    plt.legend(['PID', 'ADRC'])

    plt.figure('Control')
    plt.plot(time, u)
    plt.legend(['PID', 'ADRC'])

    plt.show()
