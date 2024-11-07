import numpy as np


def rk4_integrate(deriv, initial_state, t_span, dt, inputs):
    times = t_span
    states = [initial_state]
    state = initial_state

    for t, light in zip(times, inputs):
        k1 = dt * deriv(state, light)
        k2 = dt * deriv(state + 0.5 * k1, light)
        k3 = dt * deriv(state + 0.5 * k2, light)
        k4 = dt * deriv(state + k3, light)

        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        states.append(state)

    return times, np.array(states)


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


def deriv(state, light):
    R = state[0, ...]
    Psi = state[1, ...]
    n = state[2, ...]
    mel = state[3, ...]  # pineal melatonin

    alpha_0_func = alpha_0 * pow(light, p) / (pow(light, p) + I0)
    Bhat = G * (1.0 - n) * alpha_0_func
    LightAmp = A1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * np.cos(
        Psi + BetaL1) + A2 * 0.5 * Bhat * R * (1.0 - pow(R, 8.0)) * np.cos(
        2.0 * Psi + BetaL2)
    LightPhase = sigma * Bhat - A1 * Bhat * 0.5 * (
            pow(R, 3.0) + 1.0 / R) * np.sin(Psi + BetaL1) - A2 * Bhat * 0.5 * (
                         1.0 + pow(R, 8.0)) * np.sin(2.0 * Psi + BetaL2)

    mel_growth_rate = 0
    mel_decay = 0.8
    phi_on = np.pi - 3 * 2 * np.pi / 24
    phi_off = phi_on + 2 * np.pi * 6 / 24

    dydt = np.zeros_like(state)

    dydt[0, ...] = -1.0 * gamma * R + K * np.cos(Beta1) / 2.0 * R * (
            1.0 - pow(R, 4.0)) + LightAmp
    dydt[1, ...] = 2 * np.pi / tau + K / 2.0 * np.sin(Beta1) * (
            1 + pow(R, 4.0)) + LightPhase
    dydt[2, ...] = 60.0 * (alpha_0_func * (1.0 - n) - delta * n)

    if phi_on < np.mod(Psi, 2 * np.pi) < phi_off:
        mel_growth_rate = 0.1
    dydt[3, ...] = mel_growth_rate - mel_decay * mel

    return dydt
