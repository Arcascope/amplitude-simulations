import time

import pandas as pd
from astral.sun import sun
from astral import LocationInfo
from datetime import datetime, timedelta
import pytz
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from simple_melatonin import rk4_integrate, deriv

plt.rcParams['font.family'] = 'Arial'
standard_time_color_hex = "#FFD2A1"
dst_color_hex = "#D1C4E9"

alpha = 0.7
standard_time_color = mcolors.to_rgba(standard_time_color_hex, alpha=alpha)
dst_color = mcolors.to_rgba(dst_color_hex, alpha=alpha)
timezone_str = "America/New_York"


def scale_lux(lux):
    return -1.5 + (np.log10(lux + 1) / 5)


def save_day_plot(day_data,
                  melatonin_dst_day,
                  lux_array_day_dst,
                  melatonin_st_day,
                  lux_array_day_st,
                  day_num, timezone):
    plt.figure(figsize=(10, 6))

    # Convert day_data.index to America/New_York timezone
    day_data.index = day_data.index.tz_convert(timezone)

    scalar = 25  # Scalar for melatonin for graph

    plt.fill_between(day_data.index, scalar * melatonin_st_day, label="Current melatonin",
                     color=standard_time_color, alpha=0.55)

    plt.fill_between(day_data.index, scalar * melatonin_dst_day, label="Melatonin on pDST",
                     color=dst_color, alpha=0.55)

    plt.plot(day_data.index, scale_lux(lux_array_day_dst), label="Light Schedule (pDST)", color=dst_color)
    plt.plot(day_data.index, scale_lux(lux_array_day_st), label="Current Light Schedule", color=standard_time_color)

    mel_fraction = melatonin_dst_day[19 * 60] / melatonin_st_day[19 * 60] * 100 - 100

    day_string = f"{day_data.index[0].strftime('%b. %-d %Y')}"
    mel_string = f"\n{int(mel_fraction)}% more melatonin at wake-up"
    _ = plt.text(
        0.5, 1.1, day_string, ha='center', va='center',
        fontsize=30, fontname="Arial", transform=plt.gca().transAxes
    )
    _ = plt.text(
        0.5, 1.07, mel_string, ha='center', va='center',
        fontsize=20, fontname="Arial", transform=plt.gca().transAxes
    )
    plt.subplots_adjust(top=0.75)

    plt.xlabel("Current Local Time", fontname="Arial" if "Arial" in plt.rcParams['font.sans-serif'] else "Helvetica",
               fontsize=14)
    plt.xlim([day_data.index[0], day_data.index[0] + timedelta(hours=24)])  # Ensure x-axis covers one full day
    plt.ylim([-2, 4])
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone))

    plt.yticks([])
    plt.ylabel('')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().get_yaxis().set_ticks([])  # Removes y-axis ticks

    bedtime = day_data.index[0].replace(hour=23, minute=0)
    alarm_time = day_data.index[0].replace(hour=7, minute=0) + timedelta(days=1)

    plt.axvline(x=bedtime, color='purple', linestyle='--')
    plt.axvline(x=alarm_time, color='green', linestyle='--')

    plt.axvline(x=bedtime, color='purple', linestyle='--')
    plt.axvline(x=alarm_time, color='green', linestyle='--')

    plt.text(bedtime, 3.4, 'Bedtime', rotation=90, horizontalalignment="right", verticalalignment='center',
             color='purple', fontsize=16)
    plt.text(alarm_time, 3.4, 'Alarm', rotation=90, horizontalalignment="right", verticalalignment='center',
             color='green', fontsize=16)
    plt.xticks(fontsize=14)

    plt.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"output/{day_num:03d}.png")
    plt.close()


if __name__ == "__main__":

    start_run = time.time()  # Record the start time

    # Set the location for Boston with Longitude and Latitude
    boston = LocationInfo("Boston", "USA", "America/New_York", latitude=42.3601, longitude=-71.0589)
    miami = LocationInfo("Miami", "USA", "America/New_York", latitude=25.761681, longitude=-80.191788)

    location = miami  # miami, boston are two options
    timezone = pytz.timezone(timezone_str)

    # Define the date range of interest
    start_date = datetime(2023, 4, 1)  # Example start date
    end_date = datetime(2024, 3, 5)  # Example end date

    # Set the start date to noon in the specified timezone
    start_date = timezone.localize(datetime.combine(start_date, datetime.min.time()))
    end_date = timezone.localize(datetime.combine(end_date, datetime.min.time()))
    start_date_noon = start_date + timedelta(hours=12)

    # Holder for DST flag
    results = {"False": {}, "True": {}}

    index_to_remove = -1
    work_lux = 300
    home_lux = 100
    daylight_lux = 10000

    for permanent_dst in [False, True]:

        # Find the DST transition date (first Sunday in November for the U.S.)
        dst_transition_date = timezone.localize(datetime(start_date.year, 11, 1))
        while dst_transition_date.weekday() != 6:  # Find the first Sunday
            dst_transition_date += timedelta(days=1)
        dst_transition_date = dst_transition_date + timedelta(hours=2)

        lux_data = pd.DataFrame()

        # Iterate over each day in the date range
        for single_date in pd.date_range(start_date, end_date):
            # Get sunrise and sunset times for the day in the location's time zone
            s = sun(location.observer, date=single_date, tzinfo=timezone)
            sunrise = s['sunrise']
            sunset = s['sunset']
            label = "DST" if permanent_dst else "ST"

            # Compare the timezone-aware datetime objects
            # Apply permanent DST adjustment after the transition date
            if permanent_dst and (single_date + timedelta(hours=15)) >= dst_transition_date:
                sunrise += timedelta(hours=1)
                sunset += timedelta(hours=1)

            print(f"{label} {sunrise} {sunset}")

            # Create a time series from midnight to 11:59 pm at 1-minute intervals
            day_series = pd.date_range(single_date,
                                       single_date + timedelta(days=1) - timedelta(minutes=1),
                                       freq='1min',
                                       tz=timezone)

            if (single_date + timedelta(hours=2)).to_pydatetime() == dst_transition_date:
                day_series = pd.date_range(single_date,
                                           single_date + timedelta(days=1, hours=1) - timedelta(minutes=1),
                                           freq='1min',
                                           tz=timezone)

                index_to_remove = np.shape(lux_data)[0] + 2 * 60
            lux_values = []

            # Apply lux rules based on the time of day and sunlight presence
            for current_time in day_series:
                if datetime.strptime("00:00",
                                     "%H:%M").time() <= current_time.time() < datetime.strptime("07:00",
                                                                                                "%H:%M").time():
                    lux_values.append(0)
                elif datetime.strptime("07:00",
                                       "%H:%M").time() <= current_time.time() < datetime.strptime("09:00",
                                                                                                  "%H:%M").time():
                    if sunrise <= current_time < sunset:
                        lux_values.append(daylight_lux)
                    else:
                        lux_values.append(home_lux)
                elif datetime.strptime("09:00",
                                       "%H:%M").time() <= current_time.time() < datetime.strptime("17:00",
                                                                                                  "%H:%M").time():
                    lux_values.append(work_lux)
                elif datetime.strptime("17:00",
                                       "%H:%M").time() <= current_time.time() < datetime.strptime("22:59",
                                                                                                  "%H:%M").time():
                    if sunrise <= current_time < sunset:
                        lux_values.append(daylight_lux)
                    else:
                        lux_values.append(home_lux)
                else:
                    lux_values.append(0)

            day_df = pd.DataFrame({"DateTime": day_series, "Lux": lux_values})
            lux_data = pd.concat([lux_data, day_df])

        # Leaving this here for debugging
        lux_data.reset_index(drop=True, inplace=True)
        lux_data.head()
        lux_data.set_index("DateTime", inplace=True)

        lux_data = lux_data.reset_index()

        # Get the start time (first entry in the DateTime column)
        start_time = lux_data["DateTime"].iloc[0]

        # Calculate hours since start for each timestamp
        hours_since_start = (lux_data["DateTime"] - start_time).dt.total_seconds() / 3600

        # Convert to numpy arrays
        time_array = np.array(hours_since_start)
        lux_array = lux_data["Lux"].to_numpy()
        if permanent_dst:
            time_array = time_array[:-60]
            lux_array = np.delete(lux_array, list(range(index_to_remove, index_to_remove + 60)))

        time_array = time_array.astype(float)
        lux_array = lux_array.astype(float)

        dt = time_array[1] - time_array[0]
        initial_state = np.array([1, np.pi, 0.519, 0.0])

        times, states = rk4_integrate(deriv, initial_state, time_array, dt, lux_array)
        # output = np.mod(states[:, 1], 2 * np.pi)  # For debugging
        output = states[:, 3]

        # Smoothing for our simple model of melatonin; same effect could be achieved with another compartment
        sigma = 40  # Adjust this for more or less smoothing
        smoothed_output = gaussian_filter1d(output, sigma=sigma)
        results[str(permanent_dst)] = {
            "times": times,
            "smoothed_output": smoothed_output,
            "lux_data": lux_data,
            "lux_array": lux_array
        }

    plt.close()
    plt.plot(results["True"]["smoothed_output"][::5])
    plt.plot(results["False"]["smoothed_output"][::5])
    # plt.show()  # For debugging.
    plt.close()

    day_num = 1
    has_reached_dst_day = False
    for single_date in pd.date_range(start_date_noon, end_date):
        times = results["False"]["lux_data"]
        lux_data_dst = results["True"]["lux_array"]
        lux_data_st = results["False"]["lux_array"]
        melatonin_dst = results["True"]["smoothed_output"]
        melatonin_st = results["False"]["smoothed_output"]

        is_dst_day = False
        dst_delta = timedelta(hours=24)
        if (single_date + timedelta(hours=14)).to_pydatetime() == dst_transition_date:
            dst_delta = timedelta(hours=25)
            is_dst_day = True

        day_data = times[
            (times["DateTime"] >= single_date) & (times["DateTime"] < single_date + dst_delta)]

        melatonin_st_day = melatonin_st[day_data.index[0]:day_data.index[-1] + 1]
        lux_array_day_st = lux_data_st[day_data.index[0]:day_data.index[-1] + 1]

        offset = 0 if not has_reached_dst_day else 60

        # Drop the repeated hour
        if is_dst_day:
            repeated_hour_start = 14 * 60
            indices_to_remove = list(range(repeated_hour_start, repeated_hour_start + 60))
            melatonin_st_day = np.delete(melatonin_st_day, indices_to_remove)
            lux_array_day_st = np.delete(lux_array_day_st, indices_to_remove)
            has_reached_dst_day = True

        day_data = times[
            (times["DateTime"] >= single_date) & (times["DateTime"] < single_date + timedelta(hours=24))]

        end_index = day_data.index[0] + 60 * 24 - offset
        melatonin_dst_day = melatonin_dst[day_data.index[0] - offset:end_index]
        lux_array_day_dst = lux_data_dst[day_data.index[0] - offset:end_index]

        day_data.set_index("DateTime", inplace=True)

        print(day_num)
        save_day_plot(day_data,
                      melatonin_dst_day,
                      lux_array_day_dst,
                      melatonin_st_day,
                      lux_array_day_st,
                      day_num,
                      timezone)
        day_num += 1
