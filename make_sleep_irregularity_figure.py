import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import matplotlib
from lco import integrate_model, forger_model, hannay_model
import matplotlib.image as mpimg

hours_per_day = 24
# red for negative, gray for 0, blue for positive, light gray for zero
colors = ["red", [0.8, 0.8, 0.8], "blue"]
position = [0, 0.5, 1]

custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", list(zip(position, colors)))


def calculate_sleep_regularity_index(v, timestep):
    """
    Calculate the sleep regularity index for a given sleep-wake vector.
    The sleep regularity index, here, is defined as the probability that v(t) = v(t + 24) for all t,
    such that t + 24 is still in the range of the vector.

    Parameters:
    - v: numpy array representing the sleep-wake schedule, where 0 = wake and 1 = sleep.

    Returns:
    - Sleep regularity index: A value between 0 and 1 indicating the regularity of sleep.
    """
    matches = 0
    comparisons = 0

    offset = int(hours_per_day / timestep)

    for i in range(len(v) - offset):
        if v[i] == v[i + offset]:
            matches += 1
        comparisons += 1

    if comparisons > 0:
        regularity_index = matches / comparisons
    else:
        regularity_index = 0  # Default to 0 if no valid comparisons

    return regularity_index


def generate_sleep_schedule(simulation_days=14, dt=0.1, target_regularity=0.8, epsilon=0.01, want_fragmentation=False):
    total_sleep_hours = 8
    shift_radius = 12
    num_timesteps = int(hours_per_day / dt * simulation_days)

    variability = 0.5  # Controls variability in both shift and fragmentation
    max_iterations = 20
    iteration = 0
    delta = 0.25

    # Return the base schedule if prescribed_regularity = 1
    if target_regularity == 1:
        variability = 0

    while True:
        current_schedule = np.zeros(num_timesteps, dtype=int)
        default_bedtime = 22  # 10 PM
        time = []

        for day in range(simulation_days):
            day_start_idx = int(day * hours_per_day / dt)

            # Apply variability based on regularity_slider
            bedtime_shift = np.random.uniform(-shift_radius *
                                              variability, shift_radius * variability)
            bedtime = int((default_bedtime + bedtime_shift) % hours_per_day)
            waketime = (bedtime + total_sleep_hours) % hours_per_day

            for timestep in np.arange(0, hours_per_day, dt):
                idx = round(day_start_idx + timestep / dt)
                time.append(timestep + day * hours_per_day)
                if bedtime < waketime:
                    if bedtime <= timestep < waketime:
                        current_schedule[idx] = 1  # Sleep
                else:
                    if timestep >= bedtime or timestep < waketime:
                        current_schedule[idx] = 1  # Sleep

            # OPTIONAL: Introduce fragmentation based on regularity_slider
            if want_fragmentation:
                if np.random.rand() < variability:
                    fragmentation_length_hours = int(np.random.uniform(2, 4))
                    start_fragmentation_hour = int(np.random.uniform(bedtime,
                                                                     bedtime + total_sleep_hours - fragmentation_length_hours) % hours_per_day)
                    start_fragmentation_idx = int(
                        day_start_idx + start_fragmentation_hour / dt)

                    for hour in np.arange(0, fragmentation_length_hours, dt):
                        idx = int(start_fragmentation_idx + hour / dt)
                        if idx < len(current_schedule):  # Ensure idx is within bounds
                            current_schedule[idx] = 0  # Wake

        # Calculate the sleep regularity index for the generated schedule
        sri = calculate_sleep_regularity_index(current_schedule, timestep=dt)
        print(f"SRI: {sri}")
        print(f"Variability: {variability}")

        # Check if SRI is within the target range
        if target_regularity - epsilon <= sri <= target_regularity + epsilon:
            break  # Exit the loop if within target range

        # Adjust variability using binary search
        if sri < target_regularity:
            variability = variability - delta
        else:
            variability = variability + delta

        delta = delta / 2

        iteration += 1
        if iteration >= max_iterations:
            variability = 0.5
            iteration = 0
            delta = 0.25

    return np.array(time), current_schedule


def plot_actogram_double_plotted(sleep_wake_vector, amplitude_delta, simulation_days=14, timestep=1.0, plot_title='',
                                 is_amplitude=True):
    sleep_wake_vector = sleep_wake_vector * 1.0
    sleep_wake_vector[sleep_wake_vector == 1.0] = np.nan
    data = sleep_wake_vector.reshape(
        (simulation_days, int(hours_per_day / timestep)))

    amplitude_delta = np.insert(amplitude_delta, 0, 0)
    amplitude_delta = amplitude_delta.reshape(-1, 1)

    scaled_data = amplitude_delta.reshape(
        (simulation_days, int(hours_per_day / timestep))) * (1 + data)

    double_plotted_data = np.zeros(
        (simulation_days, int(hours_per_day / timestep * 2)))

    # Last day does not have a "next day" to concatenate
    for day in range(simulation_days - 1):
        double_plotted_data[day] = np.concatenate(
            (scaled_data[day], scaled_data[day + 1]))
    double_plotted_data[-1] = np.concatenate(
        (scaled_data[-1], np.nan * np.ones_like(scaled_data[-1])))

    fig, ax = plt.subplots(figsize=(14, 7))
    custom_cmap.set_bad(color='white')  # Set NaNs to white

    if is_amplitude:
        norm = Normalize(vmin=-10,
                         vmax=10)
    else:
        norm = Normalize(vmin=-0.15,
                         vmax=0.15)

    norm.autoscale_None([np.nan])  # Auto-scale to include NaN

    cax = ax.imshow(double_plotted_data, aspect='auto',
                    cmap=custom_cmap, norm=norm)

    font_size = 40
    tick_font_size = 36

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=tick_font_size)
    cbar.ax.get_yaxis().labelpad = 25
    if is_amplitude:
        cbar.ax.set_ylabel('%$\Delta R$', rotation=270, fontsize=30)
    else:
        cbar.ax.set_ylabel('$\Delta \phi$', rotation=270, fontsize=30)

    # Adjust ticks for 48-hour x-axis
    dt_plot = 4
    x_ticks = np.arange(0, hours_per_day / timestep * 2, dt_plot / timestep)

    x_tick_labels = [str(int(x % hours_per_day))
                     for x in np.arange(0, hours_per_day * 2, dt_plot)]
    ax.set_xlabel('Local time', fontsize=font_size)
    ax.set_ylabel('Day', fontsize=font_size)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=tick_font_size)

    day_step = 5
    ax.set_yticks(np.arange(0, simulation_days, day_step))
    ax.set_yticklabels(np.arange(0, simulation_days, day_step),
                       fontsize=tick_font_size)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"outputs/{plot_title}.png", dpi=300)
    plt.close()
    # plt.show()


def amplitude_derivative_cartesian(state_vector, d_state_vector):
    x = state_vector[0]
    dx = d_state_vector[0]
    y = state_vector[1]
    dy = d_state_vector[1]
    return (x * dx + y * dy) / np.sqrt(x * x + y * y)


def phase_derivative_cartesian(state_vector, d_state_vector):
    x = state_vector[0]
    dx = d_state_vector[0]
    y = state_vector[1]
    dy = d_state_vector[1]

    denominator = (x * x + y * y)
    return -x * dy / denominator + y * dx / denominator


if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = 'Arial'

    num_days = 24
    regularity = 0.8
    dt = 0.1
    light_scalar = 500

    for model in ['forger', 'hannay']:
        for prescribed_regularity in [0.6, 0.75, 0.85, 0.95, 1.0]:
            timestamps, schedule = generate_sleep_schedule(simulation_days=num_days,
                                                           target_regularity=prescribed_regularity,
                                                           dt=dt)

            print(f"Target SRI: {prescribed_regularity}")
            print(
                f"Actual SRI: {calculate_sleep_regularity_index(schedule, dt)}")

            # Since 1 = sleep, need to flip 0 and 1:
            light = (1 - schedule) * light_scalar
            amplitude_change_percent = []
            phase_change = []

            if model == 'forger':
                initial_condition = np.array(
                    [-0.6717444, -0.85167686, 0.15397873])

                sol = integrate_model(timestamps,
                                      light,
                                      initial_condition,
                                      model)

                # Calculate dR using chain rule
                for i in range(np.shape(sol)[1] - 1):
                    state = sol[:, i]
                    d_state_light = forger_model(sol[:, i], light[i])
                    dR = amplitude_derivative_cartesian(state, d_state_light)
                    dPsi = phase_derivative_cartesian(state, d_state_light)

                    d_state_dark = forger_model(sol[:, i], 0)
                    dR_dark = amplitude_derivative_cartesian(
                        state, d_state_dark)
                    dPsi_dark = phase_derivative_cartesian(state, d_state_dark)

                    # Will be zero in the dark
                    R = np.sqrt(state[0] * state[0] + state[1] * state[1])
                    psi = np.arctan2(state[1], state[0])
                    amplitude_change_percent.append((dR - dR_dark) / R * 100)
                    phase_change.append(dPsi - dPsi_dark)

            if model == 'hannay':
                initial_condition = np.array(
                    [0.83656626, 146.6791648, 0.3335272])

                sol = integrate_model(timestamps,
                                      light,
                                      initial_condition,
                                      model)

                # First parameter is R, no need for calculus
                for i in range(np.shape(sol)[1] - 1):
                    state = sol[:, i]
                    d_state_light = hannay_model(sol[:, i], light[i])
                    dR = d_state_light[0]
                    dPsi = d_state_light[1]

                    d_state_dark = hannay_model(sol[:, i], 0)
                    dR_dark = d_state_dark[0]
                    dPsi_dark = d_state_dark[1]

                    R = state[0]
                    psi = np.mod(state[1], 2 * np.pi)

                    # Will be zero in the dark
                    amplitude_change_percent.append((dR - dR_dark) / R * 100)
                    phase_change.append(dPsi - dPsi_dark)

            title = f"{model}_{prescribed_regularity}_amplitude"
            plot_actogram_double_plotted(schedule,
                                         amplitude_change_percent,
                                         timestep=dt,
                                         simulation_days=num_days,
                                         plot_title=title)

            title = f"{model}_{prescribed_regularity}_phase"
            plot_actogram_double_plotted(schedule,
                                         phase_change,
                                         timestep=dt,
                                         simulation_days=num_days,
                                         plot_title=title,
                                         is_amplitude=False)

    imgA = mpimg.imread('outputs/forger_1.0_amplitude.png')
    imgB = mpimg.imread('outputs/forger_0.75_amplitude.png')

    imgC = mpimg.imread('outputs/hannay_1.0_amplitude.png')
    imgD = mpimg.imread('outputs/hannay_0.75_amplitude.png')

    # Create a figure with 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 5))

    # Place each image on the grid and add labels
    images = [(imgA, "A"), (imgB, "B"), (imgC, "C"), (imgD, "D")]
    for ax, (img, label) in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        ax.text(0.05, 1.05, label, transform=ax.transAxes, fontsize=20, color="black",
                fontweight="bold")
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig("outputs/figure_sleep_regularity.png", dpi=500, bbox_inches='tight')
    plt.show()
