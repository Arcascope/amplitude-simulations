import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# Define helper functions (VTRPin, VTPH, VAADC, HTcatab, AANT, VAANT, VHOMT, HOMT)
# These must match your existing implementations

def VTRPin(b, sc):
    k1 = 10 * 900 * sc
    k2 = 330
    return k1 * b / (k2 + b)


def VTPH(b, sc):
    k1 = 40
    k4 = 1000
    k5 = 2.5 * 278 * sc
    return k5 * b / (k1 + b + b ** 2 / k4)


def VPOOL(b, c, sc):
    # Pool dynamics calculation
    a = (10 * b - 1 * c) * sc

    return a


def VAADC(b, sc):
    k1 = 160
    V = 2.5 * 400 * sc
    return V * b / (k1 + b)


def HTcatab(b, sc):
    k1 = 95
    k2 = 1.42 * 400
    return k2 * b / (k1 + b)


def AANT(t):
    t = np.mod(t, 24)
    if t < 8:
        return 0.5
    elif t < 18:
        return 0.5 + 14 * (t - 8) ** 2 / (15 + (t - 8) ** 2)
    else:
        return 0.5 + 12 * np.exp(-10 * (t - 18))


def VAANT(c, sc):
    k1 = 1235
    k2 = 0.5 * 250
    return k2 * c / (k1 + c) * sc


def VHOMT(c, sc):
    k1 = 40
    k2 = 3 * 1.4 * 80
    return k2 * c / (k1 + c) * sc


def HOMT(t):
    t = np.mod(t, 24)
    if t < 8:
        return 0.5
    elif t < 18:
        return 0.5 + 0.15 * (t - 8) / (2 + (t - 8))
    else:
        return 0.5


# Define the ODE system
def msc(t, y, sc):
    """
    Define the ODE system for melatonin and serotonin pathways.

    Inputs:
        t: float, current time
        y: array-like, state variables
        sc: float, scaling factor
    Output:
        dy: array-like, derivatives of state variables
    """
    dy = np.zeros(13)  # Initialize derivatives array

    # Equations
    dy[0] = VTRPin(y[10], sc) - VPOOL(y[0], y[1], sc) - VTPH(y[0], sc) - 1 * y[0] * sc
    dy[1] = VPOOL(y[0], y[1], sc) - 2 * y[1]
    dy[2] = VTPH(y[0], sc) - VAADC(y[2], sc)
    dy[3] = VAADC(y[2], sc) - HTcatab(y[3], sc) - AANT(t) * VAANT(y[3], sc)
    dy[4] = 0
    dy[5] = AANT(t) * VAANT(y[3], sc) - HOMT(t) * VHOMT(y[5], sc) - 1 * y[5]
    dy[6] = 0
    dy[7] = HOMT(t) * VHOMT(y[5], sc) - 2.2 * y[7] + (15000) * 1 * y[11] - (0.01) * (2.2 * y[7] + 500 * y[12]) - 2.3 * \
            y[7]
    dy[8] = 0
    dy[9] = np.sin(t)
    dy[10] = 0
    dy[11] = (2.2 / 15000) * y[7] - 1 * y[11] - 7 * y[11]
    dy[12] = (0.01) * (2.2 / 500 * y[7] - 1 * y[12]) - 7 * y[12]

    return dy


# Main function to run the simulation and generate figures
def generate_figure():
    # Simulation parameters
    time = 24
    sc = 1  # Scaling factor
    y0 = [200, 666, 157, 498, 0, 3.95, 0, 3.4, 0, 0, 96.00, 0.0000620, 0.000005]

    # Solve the ODE
    sol = solve_ivp(
        lambda t, y: msc(t, y, sc),
        [0, time],
        y0,
        t_eval=np.linspace(0, time, 5000),
        method='RK45'
    )

    T = sol.t
    Y = sol.y

    # Plot the results
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 3, 1)
    plt.plot(T, Y[0, :], 'b', label='ctrp')
    plt.plot(T, Y[1, :], 'g', label='trppool')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 2)
    plt.plot(T, Y[2, :], 'g', label='c5HTp')
    plt.plot(T, Y[3, :], 'b', label='c5HT')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 3)
    plt.plot(T, VTPH(Y[0, :], sc), 'g', label='VTPH')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 4)
    plt.plot(T, [AANT(t) for t in T], 'b', label='AANT')
    plt.plot(T, [HOMT(t) * 10 for t in T], 'r', label='10*HOMT')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 5)
    plt.plot(T, [VAANT(y, sc) for y in Y[3, :]], 'b', label='VAANT')
    plt.plot(T, [VHOMT(y, sc) for y in Y[5, :]], 'g', label='VHOMT')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 6)
    plt.plot(T, Y[5, :], 'r', label='ca5ht')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 7)
    mult = [AANT(t) * VAANT(y, sc) for t, y in zip(T, Y[3, :])]
    mult2 = [HOMT(t) * VHOMT(y, sc) for t, y in zip(T, Y[5, :])]
    plt.plot(T, mult, 'b', label='A*VA')
    plt.plot(T, mult2, 'g', label='H*VH')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 8)
    plt.plot(T, 10 ** 6 * Y[11, :], 'm', label='bmel-pM')
    plt.plot(T, 10 ** 6 * Y[12, :], 'g', label='CSFmel')
    plt.legend(fontsize=12)

    plt.subplot(3, 3, 9)
    plt.plot(T, Y[7, :], 'm', label='cmel')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_figure()
