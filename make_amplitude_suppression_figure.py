import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from lco import integrate_model


def generate_light_exposure_vector(duration, dt, lights_on_time, lights_off_time, light_level=500, spread=0.0):
    time_vector = np.arange(0, duration + dt, dt)

    light_exposure = []
    debug_wakes = []
    debug_beds = []

    for time in time_vector:
        if np.mod(time, 24) == 0:
            offset_for_day = (np.random.rand(1)[0] * spread * 2)
            effective_lights_on = np.mod(lights_on_time + offset_for_day, 24)
            effective_lights_off = np.mod(lights_off_time + offset_for_day, 24)
            debug_wakes.append(effective_lights_on)
            debug_beds.append(effective_lights_off)

        if effective_lights_on <= effective_lights_off:
            if effective_lights_on <= np.mod(time, 24) <= effective_lights_off:
                light_exposure.append(light_level)  # Lights on
            else:
                light_exposure.append(0)  # Darkness
        else:
            if effective_lights_on <= np.mod(time, 24) or np.mod(time, 24) <= effective_lights_off:
                light_exposure.append(light_level)  # Lights on
            else:
                light_exposure.append(0)  # Darkness

    return time_vector, np.array(light_exposure)


if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = 'Arial'

    plt.close()
    spread_range = list(range(1, 8))

    # TODO: Replace with circadian package
    for model in ['forger', 'hannay']:
        average_amplitude = []

        for spread_value in spread_range:
            # Random IC; will wash out; works for either model
            initial_condition = np.array([1.0, 0, 0.1])

            num_days = 500
            max_time = 24 * num_days  # 24 hours for one day
            step_size = 0.1  # 0.1 hour step size
            lights_on = 8
            lights_off = 24

            timestamps, light_exposure_vector = generate_light_exposure_vector(max_time,
                                                                               step_size,
                                                                               lights_on,
                                                                               lights_off,
                                                                               light_level=1000,
                                                                               spread=spread_value / 2)

            sol = integrate_model(timestamps,
                                  light_exposure_vector,
                                  initial_condition,
                                  model)
            if model == 'forger':
                amplitude = np.sqrt(
                    sol[0, :] * sol[0, :] + sol[1, :] * sol[1, :])
            else:
                amplitude = sol[0, :]
            sol_len = len(sol)
            average_amplitude.append(np.mean(amplitude[sol_len // 2:]))

        plt.plot(spread_range, average_amplitude, 'ko')
        title_font_size = 28
        font_size = 22
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plt.title("Amplitude and sleep variability", fontsize=title_font_size)

        plt.xlabel("Spread in wake and bed time", fontsize=font_size)
        plt.ylabel("Average model amplitude", fontsize=font_size)
        plt.savefig(f"outputs/{model}_spread_and_amplitude.png", dpi=300)
        plt.close()
