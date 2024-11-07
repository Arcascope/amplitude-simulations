import time

from simple_melatonin import rk4_integrate, deriv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
dt = 0.05


def generate_light_dark_schedule(light_hours, period, brightness, dt=0.1, cycles_needed=200):
    # Calculate total hours in a cycle and the number of cycles
    dark_hours = period - light_hours
    cycle_duration = light_hours + dark_hours

    # Calculate steps per phase
    light_steps = int(light_hours / dt)
    dark_steps = int(dark_hours / dt)

    # Define one cycle with specified light and dark periods
    light_phase = np.full(light_steps, brightness)
    dark_phase = np.zeros(dark_steps)
    one_cycle = np.concatenate((light_phase, dark_phase))

    # Repeat the cycle 50 times
    schedule = np.tile(one_cycle, cycles_needed)
    time_array = np.arange(0, cycle_duration * cycles_needed, dt)

    return time_array, schedule


if __name__ == "__main__":

    start_clock = time.time()
    for WANT_AVERAGE in [True, False]:
        light_hours_range = range(1, 25)
        brightness_levels = [0, 50, 150, 500, 1000, 10000]

        output_array = np.zeros((len(light_hours_range), len(brightness_levels)))

        for i, light_hours in enumerate(light_hours_range):
            print(f"Running {light_hours}hrs of light...")

            for j, brightness in enumerate(brightness_levels):
                time_array, schedule = generate_light_dark_schedule(light_hours, 24, brightness, dt=dt)
                initial_state = np.array([1, np.pi, 0.519, 0.0])

                times, states = rk4_integrate(deriv, initial_state, time_array, time_array[1] - time_array[0], schedule)
                cut_point = int(len(states) / 2)
                if WANT_AVERAGE:
                    output = np.mean(states[cut_point:, 0])
                else:
                    output = np.max(states[cut_point:, 0])

                plt.close()
                plt.plot(times, states[1:, 0])
                plt.title(f"{light_hours} of light, {brightness} lux")
                plt.savefig(f"debug/light_{light_hours}_brightness_{brightness}")
                plt.close()

                output_array[i, j] = output

        X, Y = np.meshgrid(range(len(brightness_levels)), light_hours_range)  # Use index positions for brightness levels
        plt.contourf(X, Y, output_array, cmap="viridis")
        plt.colorbar(label="Model amplitude")

        plt.xlabel("Brightness")
        plt.ylabel("Light Duration (hours)")
        plt.xticks(ticks=range(len(brightness_levels)), labels=brightness_levels)

        metric = "Average" if WANT_AVERAGE else "Max"
        plt.title(f"Brightness and light duration for a 24 hr day\nEffects on {metric} Amplitude")
        plt.savefig(f"Brightness and light duration on a 24 hours day - Effects on {metric} Amplitude.png", dpi=300)
        plt.close()

        period_duration = [4, 8, 16, 20, 24, 28, 32, 36, 40]
        brightness_levels = [0, 50, 150, 500, 1000, 10000]

        output_array = np.zeros((len(period_duration), len(brightness_levels)))

        for i, period in enumerate(period_duration):
            print(f"Running period {period}...")

            for j, brightness in enumerate(brightness_levels):
                time_array, schedule = generate_light_dark_schedule(period / 2, period, brightness, dt=dt,
                                                              cycles_needed=int(50 * 24 / period))
                initial_state = np.array([1, np.pi, 0.519, 0.0])

                times, states = rk4_integrate(deriv, initial_state, time_array, time_array[1] - time_array[0], schedule)
                cut_point = int(len(states) / 2)  # Trim ICs

                plt.close()
                plt.plot(times, states[1:, 0])
                plt.title(f"{period} hr long period, {brightness} lux")
                plt.savefig(f"debug/period_{period}_brightness_{brightness}")
                plt.close()

                if WANT_AVERAGE:
                    output = np.mean(states[cut_point:, 0])
                else:
                    output = np.max(states[cut_point:, 0])
                output_array[i, j] = output

        X, Y = np.meshgrid(range(len(brightness_levels)), period_duration)
        plt.contourf(X, Y, output_array, cmap="viridis")
        plt.colorbar(label="Model amplitude")

        plt.xlabel("Brightness")
        plt.xticks(ticks=range(len(brightness_levels)), labels=brightness_levels)

        plt.ylabel("Period Duration (hours)")
        metric = "Average" if WANT_AVERAGE else "Max"
        plt.title(f"Brightness and day length (equal L and D)\nEffects on {metric} Amplitude")
        plt.savefig(f"Brightness and day length (equal L and D) - Effects on {metric} Amplitude.png", dpi=300)

        # plt.show()
    end_clock = time.time()
    elapsed_time = end_clock - start_clock
    print(f"Function took {elapsed_time / 60 :.4f} minutes")