import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# Define the Gaussian function
def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(13.5, 4))

    first_position = -4
    offset = 25
    start = -10
    width = 20

    mu = 0
    sigma = 2
    x = np.linspace(-7, 7, 1000)
    y1 = gaussian(x, mu, sigma)  # First Gaussian

    sigma_2 = 1.5
    peak_difference = 0.8
    y2 = peak_difference * 0.5 * gaussian(x, mu, sigma_2)  # Second Gaussian, vertically squished

    # Plot first Gaussian curve and fill
    ax.plot(x, y1, color='black')
    ax.fill_between(x, y1, color='skyblue', alpha=0.5)

    # Plot second Gaussian curve and fill
    ax.plot(x + offset, y2, color='black')
    ax.fill_between(x + offset, y2, color='lightcoral', alpha=0.5)

    ax.add_patch(plt.Rectangle((start, -0.02), width, -
                 0.02, color='grey', alpha=0.3, hatch='//'))
    ax.add_patch(plt.Rectangle((offset + start, -0.02), width, -
                 0.02, color='grey', alpha=0.3, hatch='//'))

    ax.text(0, -0.06, "Melatonin collection over night from individual",
            ha='center', va='top', fontsize=12)
    ax.text(offset, -0.06, "Melatonin collection over night from same person",
            ha='center', va='top', fontsize=12)

    ax.text(offset / 2, max(np.max(y1), np.max(y2))
            * 0.3, "...", fontsize=30, ha='center')
    ax.text(offset / 2, max(np.max(y1), np.max(y2)) * 0.38,
            "Time elapses", fontsize=15, ha='center')

    delta_x = x[1] - x[0]
    scalar = 200
    auc_y1 = round(scalar * (np.sum(y1) * delta_x))
    auc_y2 = round(scalar * (np.sum(y2) * delta_x))
    auc_ratio = auc_y2 / auc_y1

    threshold = 0.01
    scalar = 1 / 50
    duration_y1 = scalar * np.sum(y1 > threshold)
    duration_y2 = scalar * np.sum(y2 > threshold)

    rightmost_x = 45
    y_pos = 0.16
    rectangle_height = 0.03
    rectangle_width = 8
    ax.text(rightmost_x + 0.9 * rectangle_width, y_pos *
            0.9, "=", fontsize=20, ha='center', va='center')
    ax.text(rightmost_x + 1.3 * rectangle_width, y_pos * 0.9,
            f"{auc_ratio:.2f}", fontsize=20, ha='center', va='center')
    ax.text(rightmost_x + 0.6 * rectangle_width, y_pos * 1.4, "Relative Melatonin\nAUC Change:", fontsize=12, ha='center',
            va='center')

    ax.add_patch(
        plt.Rectangle((rightmost_x - rectangle_width / 2, y_pos), rectangle_width, rectangle_height, color='lightcoral',
                      alpha=0.5))

    ax.text(rightmost_x, y_pos + rectangle_height / 2,
            f"{auc_y2} pg/ml", fontsize=15, ha='center', va='center')

    ax.add_patch(
        plt.Rectangle((rightmost_x - rectangle_width / 2, y_pos - 2 * rectangle_height), rectangle_width,
                      rectangle_height, color='skyblue', alpha=0.5))

    ax.text(rightmost_x, y_pos - 1.5 * rectangle_height,
            f"{auc_y1} pg/ml", fontsize=15, ha='center', va='center')

    ax.plot([rightmost_x - rectangle_width / 2, rightmost_x + rectangle_width / 2],
            [y_pos - rectangle_height / 2, y_pos - rectangle_height / 2], color='black', linewidth=2)

    y_pos = -0.05
    ax.text(rightmost_x + 0.9 * rectangle_width, y_pos *
            1.3, "=", fontsize=20, ha='center', va='center')
    ax.text(rightmost_x + 1.3 * rectangle_width, y_pos * 1.3,
            f"{(duration_y2/duration_y1):.2f}", fontsize=20, ha='center', va='center')
    ax.text(rightmost_x + 0.6 * rectangle_width, 0.02, "Relative Melatonin\nDuration Change:", fontsize=12, ha='center',
            va='center')

    ax.add_patch(
        plt.Rectangle((rightmost_x - rectangle_width / 2, y_pos), rectangle_width, rectangle_height, color='lightcoral',
                      alpha=0.5))

    ax.text(rightmost_x, y_pos + rectangle_height / 2,
            f"{duration_y2:.1f} hours", fontsize=15, ha='center', va='center')

    ax.add_patch(
        plt.Rectangle((rightmost_x - rectangle_width / 2, y_pos - 2 * rectangle_height), rectangle_width,
                      rectangle_height, color='skyblue', alpha=0.5))

    ax.text(rightmost_x, y_pos - 1.5 * rectangle_height,
            f"{duration_y1:.1f} hours", fontsize=15, ha='center', va='center')

    ax.plot([rightmost_x - rectangle_width / 2, rightmost_x + rectangle_width / 2],
            [y_pos - rectangle_height / 2, y_pos - rectangle_height / 2], color='black', linewidth=2)


    ax.axis('off')
    plt.savefig("outputs/melatonin.png", dpi=300)
    plt.show()
