import matplotlib.pyplot as plt
from lco import SinglePopModel
import numpy as np


def solar_lux(hour):
    # Constants
    sunrise = 6.0  # Sunrise at 6 AM
    sunset = 18.0  # Sunset at 6 PM
    peak_lux = 5000  # Peak solar lux at noon

    if hour < sunrise or hour > sunset:
        return 0  # No sunlight before sunrise and after sunset

    # Convert hour to radians for the sinusoidal function, scaling from 0 to pi over the daylight hours
    radians_per_hour = np.pi / (sunset - sunrise)
    hour_angle = (hour - sunrise) * radians_per_hour

    # Sinusoidal function to mimic the rise and fall of the sun
    lux = peak_lux * np.sin(hour_angle)

    return lux


def phase_ic_guess(time_of_day: float):
    time_of_day = np.fmod(time_of_day, 24.0)

    # Wake at 7 am after 8 hours of sleep, state at 00:00
    psi = 1.65238233

    # Convert to radians, add to phase
    psi += time_of_day * np.pi / 12
    return psi


def generate_light_exposure_vector(max_time, step_size, lights_on, lights_off, light_level=500, std=0.0):
    timestamps = np.arange(0, max_time + step_size, step_size)

    light_exposure = []
    debug_wakes = []
    debug_beds = []
    low_light_level = 500
    daylight_level = 200
    for time in timestamps:

        # light_level = solar_lux(np.mod(time, 24))
        #
        # if light_level < low_light_level:
        #     light_level = low_light_level

        light_level = low_light_level
        # light_level = low_light_level
        if np.mod(time, 24) == 0:
            # effective_lights_on = np.mod(lights_on + (np.random.rand(1)[0] * std * 2) - std, 24)
            # effective_lights_off = np.mod(lights_off + (np.random.rand(1)[0] * std * 2) - std, 24)

            offset_for_day = (np.random.rand(1)[0] * std * 2)
            effective_lights_on = np.mod(lights_on + offset_for_day, 24)
            effective_lights_off = np.mod(lights_off + offset_for_day, 24)

            # effective_lights_off = np.mod(lights_off + np.random.rand(1)[0] * std * 2, 24)
            # debug_wakes.append(lights_on)
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
    print(debug_beds)
    # plt.close()
    # plt.hist(debug_wakes, bins=10)
    # plt.show()
    # plt.plot(timestamps, np.array(light_exposure))
    # plt.show()
    return timestamps, np.array(light_exposure)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.close()
    std_dev_range = list(range(1, 8))
    average_amplitude = []
    for std_deviation in std_dev_range:
        model = SinglePopModel()
        initial_condition = np.array([0.6, phase_ic_guess(0), 0.0])
        initial_condition = np.array([0.0, 1.0, 0.0])

        num_days = 500
        max_time = 24 * num_days  # 24 hours for one day
        step_size = 0.1  # 0.1 hour step size
        lights_on = 8
        lights_off = 24
        # Standard deviation for irregular wake and bed times

        timestamps, light_exposure_vector = generate_light_exposure_vector(max_time,
                                                                           step_size,
                                                                           lights_on,
                                                                           lights_off,
                                                                           light_level=500,
                                                                           std=std_deviation / 2)

        # plt.plot(timestamps, light_exposure_vector)
        # plt.show()

        # TODO: Replace with circadian package
        sol = model.integrate_model(timestamps,
                                    light_exposure_vector,
                                    initial_condition)
        # plt.plot(timestamps, sol[0, :])
        sol_len = len(sol)
        amplitude = np.sqrt(sol[0, :] * sol[0, :] + sol[1, :] * sol[1, :])
        average_amplitude.append(np.mean(amplitude[sol_len // 2:]))

        # plt.xlabel('Time hours since start')
        # plt.ylabel('Model output')
    plt.plot(std_dev_range, average_amplitude, 'ko')
    plt.xlabel("Spread in wake and bed time")
    plt.ylabel("Average model amplitude")
    plt.savefig("bed-wake-stabilization.png", dpi=300)
    plt.show()

    # dt = 0.1  # hours
    # days = 20
    # time = np.arange(0, 24 * days, dt)
    # regular_lux = 500
    # schedule = LightSchedule.Regular(regular_lux, lights_on=8, lights_off=24)
    # light_input = schedule(time)
    # model_list = [Forger99(), Jewett99(), Hannay19(), Hannay19TP()]
    # equilibrium_states = []
    #
    # for model in model_list:
    #     time_eq = np.arange(0, 24 * days, dt)
    #     final_state = model.equilibrate(time_eq, light_input, num_loops=2)
    #     equilibrium_states.append(final_state)
    #
    # days = 3
    # time = np.arange(0, 24 * days, dt)
    # pulse_num = 6
    # pulse_lux = 1e4
    # pulse_duration = 1  # hour
    # start_values = np.linspace(32, 47, pulse_num)
    #
    # simulation_result = {}
    #
    # for idx, model in enumerate(model_list):
    #     simulation_result[str(model)] = {}
    #     for pulse_start in start_values:
    #         schedule = LightSchedule.Regular(regular_lux, lights_on=8, lights_off=24)
    #         schedule += LightSchedule.from_pulse(pulse_lux, pulse_start, pulse_duration)
    #         light_input = schedule(time)
    #         trajectory = model(time, equilibrium_states[idx], light_input)
    #         simulation_result[str(model)][str(pulse_start)] = {
    #             'light': light_input,
    #             'trajectory': trajectory
    #         }
