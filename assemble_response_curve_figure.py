
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

if __name__ == '__main__':
    matplotlib.rcParams['font.family'] = 'Arial'

    imgA = mpimg.imread('outputs/prc_1.png')
    imgB = mpimg.imread('outputs/arc_1.png')

    imgC = mpimg.imread('outputs/prc_65.png')
    imgD = mpimg.imread('outputs/arc_65.png')

    # Create a figure with 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 7))

    # Place each image on the grid and add labels
    images = [(imgA, "A"), (imgB, "B"), (imgC, "C"), (imgD, "D")]
    for ax, (img, label) in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")  # Hide axes
        ax.text(0.05, 1.03, label, transform=ax.transAxes, fontsize=20, color="black",
                fontweight="bold")
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig("outputs/figure_response_curves.png", dpi=500, bbox_inches='tight')
    plt.show()
