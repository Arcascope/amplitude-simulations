# Re-import necessary libraries and define the Gaussian function again after the reset
import matplotlib.pyplot as plt
import numpy as np


# Define the Gaussian function
def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


if __name__ == '__main__':
    # Set up the figure and axes again
    fig, ax = plt.subplots(figsize=(12, 4))

    first_position = -4
    offset = 25
    start = -10
    width = 20

    # Parameters
    mu = 0
    sigma = 2
    x = np.linspace(-7, 7, 1000)
    y1 = gaussian(x, mu, sigma)  # First Gaussian
    y2 = 0.6 * gaussian(x, mu, sigma)  # Second Gaussian, vertically squished

    # Plot first Gaussian curve and fill
    ax.plot(x, y1, color='black')
    ax.fill_between(x, y1, color='skyblue', alpha=0.5)

    # Plot second Gaussian curve and fill
    ax.plot(x + offset, y2, color='black')
    ax.fill_between(x + offset, y2, color='lightcoral', alpha=0.5)

    # Shaded rectangles
    ax.add_patch(plt.Rectangle((start, -0.02), width, -0.02, color='grey', alpha=0.3))
    ax.add_patch(plt.Rectangle((offset + start, -0.02), width, -0.02, color='grey', alpha=0.3))

    # Labels
    ax.text(0, -0.06, "Melatonin collection over night from individual", ha='center', va='top')
    ax.text(offset, -0.06, "Melatonin collection over night from same person", ha='center', va='top')

    # Ellipses and intervention text
    ax.text(offset / 2, max(np.max(y1), np.max(y2)) * 0.3, "...", fontsize=30, ha='center')
    ax.text(offset / 2, max(np.max(y1), np.max(y2)) * 0.38, "Intervention", fontsize=12, ha='center')

    # Calculate the area under each curve for the AUC comparison visualization
    delta_x = x[1] - x[0]
    auc_y1 = np.sum(y1) * delta_x
    auc_y2 = np.sum(y2) * delta_x
    auc_ratio = auc_y2 / auc_y1

    # Position for the AUC comparison visualization
    rightmost_x = 35
    y_pos = 0.15
    rectangle_height = 0.03
    rectangle_width = 4
    ax.text(rightmost_x + 0.9 * rectangle_width, y_pos * 0.9, "=", fontsize=20, ha='center', va='center')
    ax.text(rightmost_x + 1.7 * rectangle_width, y_pos * 0.9, f"{auc_ratio:.2f}", fontsize=20, ha='center', va='center')
    ax.text(rightmost_x + rectangle_width, y_pos * 1.4, "Relative Melatonin\nAUC Change", fontsize=12, ha='center', va='center')

    # Fraction visualization
    ax.add_patch(
        plt.Rectangle((rightmost_x - rectangle_width / 2, y_pos), rectangle_width, rectangle_height, color='lightcoral',
                      alpha=0.5))
    ax.add_patch(
        plt.Rectangle((rightmost_x - rectangle_width / 2, y_pos - 2 * rectangle_height), rectangle_width,
                      rectangle_height, color='skyblue', alpha=0.5))

    ax.plot([rightmost_x - rectangle_width / 2, rightmost_x + rectangle_width / 2],
            [y_pos - rectangle_height / 2, y_pos - rectangle_height / 2], color='black', linewidth=2)

    # Remove box and axes labels
    ax.axis('off')

    plt.show()
