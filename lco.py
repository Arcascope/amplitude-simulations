from pathlib import Path
import sys
import numpy as np


def integrate_model(ts: np.ndarray, light_est: np.ndarray,
                    u0: np.ndarray = None, model='forger'):
    sol = np.zeros((3, light_est.shape[0]))
    sol[:, 0] = u0
    delta_t = np.diff(ts)
    for k in range(light_est.shape[0] - 1):
        if model == 'forger':
            u0 = forger_model_rk4_step(u0, light_est[k], dt=delta_t[k])
        else:
            u0 = hannay_model_rk4_step(u0, light_est[k], dt=delta_t[k])

        sol[:, k + 1] = u0
    return sol


def alph_forger(light):
    I0 = 9500.0
    p = .6
    a0 = .16
    return a0 * (np.power(light, p) / np.power(I0, p))


def forger_model(u, light):
    x = u[0]
    xc = u[1]
    n = u[2]

    tx = 24.2
    G = 19.875
    k = .55
    mu = .23
    b = 0.013

    Bh = G * (1 - n) * alph_forger(light)
    B = Bh * (1 - .4 * x) * (1 - .4 * xc)

    dydt = [0, 0, 0]

    dydt[0] = np.pi / 12.0 * (xc + B)
    dydt[1] = np.pi / 12.0 * (
            mu * (xc - 4.0 * np.power(xc, 3.0) / 3.0) - x * (np.power((24.0 / (.99669 * tx)), 2.0) + k * B))
    dydt[2] = 60.0 * (alph_forger(light) * (1.0 - n) - b * n)

    return np.array(dydt)


def hannay_model(u, light):
    R, Psi, n = u

    tau = 23.84
    K = 0.06358
    gamma = 0.024
    Beta1 = -0.09318
    A1 = 0.3855
    A2 = 0.1977
    BetaL1 = -0.0026
    BetaL2 = -0.957756
    sigma = 0.0400692
    G = 33.75
    alpha_0 = 0.05
    delta = 0.0075
    p = 1.5
    I0 = 9325.0

    alpha_0_func = alpha_0 * pow(light, p) / (pow(light, p) + I0)
    Bhat = G * (1.0 - n) * alpha_0_func
    LightAmp = A1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * np.cos(
        Psi + BetaL1) + A2 * 0.5 * Bhat * R * (1.0 - pow(R, 8.0)) * np.cos(
        2.0 * Psi + BetaL2)
    LightPhase = sigma * Bhat - A1 * Bhat * 0.5 * (
            pow(R, 3.0) + 1.0 / R) * np.sin(Psi + BetaL1) - A2 * Bhat * 0.5 * (
                         1.0 + pow(R, 8.0)) * np.sin(2.0 * Psi + BetaL2)

    du = np.zeros(3)
    du[0] = -1.0 * gamma * R + K * np.cos(Beta1) / 2.0 * R * (
            1.0 - pow(R, 4.0)) + LightAmp
    du[1] = 2 * np.pi / tau + K / 2.0 * np.sin(Beta1) * (
            1 + pow(R, 4.0)) + LightPhase
    du[2] = 60.0 * (alpha_0_func * (1.0 - n) - delta * n)

    return du


def forger_model_rk4_step(ustart: np.ndarray, light_val: float, dt: float):
    k1 = forger_model(ustart, light_val)
    k2 = forger_model(ustart + dt / 2 * k1, light_val)
    k3 = forger_model(ustart + dt / 2 * k2, light_val)
    k4 = forger_model(ustart + dt * k3, light_val)
    return ustart + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def hannay_model_rk4_step(ustart: np.ndarray, light_val: float, dt: float):
    k1 = hannay_model(ustart, light_val)
    k2 = hannay_model(ustart + dt / 2 * k1, light_val)
    k3 = hannay_model(ustart + dt / 2 * k2, light_val)
    k4 = hannay_model(ustart + dt * k3, light_val)
    return ustart + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
