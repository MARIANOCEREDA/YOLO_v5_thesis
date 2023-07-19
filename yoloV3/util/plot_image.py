import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config.config import config as cfg

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""

    config_data = cfg()
    cmap = plt.get_cmap("tab20b")
    class_labels = config_data["class_name"]
    
    im = np.array(image)
    height, width, _ = im.shape

    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 7, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]

        box = box[2:]

        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        point = patches.Circle((box[0] * width, box[1] * height), radius=1.5, color='yellow')
        ax.add_patch(point)

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=f"S:{box[-1]} ",
            color="white",
            verticalalignment="top",
            fontsize=6,
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    
    plt.show()
