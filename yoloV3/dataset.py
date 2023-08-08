from config.config import config as cfg
import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from util.iou import iou_width_height as iou
from util.nms import non_max_suppression as nms
from util.test_transform import test_transforms
from util.cells_to_bboxes import cells_to_bboxes
from util.plot_image import plot_image

'''(
    cells_to_bboxes,
    iou
    nms
    plot_image
)'''

config = cfg()
    
class CustomDataset(Dataset):

    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size,
        detection_layers=[13, 26, 52],
        num_classes=1,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file) # format -> [image_id, class, x_c, y_c, w,h]
        self.imgs_dir = img_dir
        self.labels_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.detection_layers = detection_layers
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index:int) -> any:
        """
        
        Parameters:
            index (int) : Index of the iteration
        
        Returns:
            image, tuple: 
                - image: The image for the current index
                - tuple of targets: If we have 3 detection layers of [13,26,52], then the result will be

                list 1 -> [1 , 13, 13, 6] -> for scale 1
                list 2 -> [1 , 26, 26, 6] -> for scale 2
                list 3 -> [1 , 52, 52, 6] -> for scale 3

        """

        # Get paths to labels and images
        image_path = os.path.join(self.imgs_dir, self.annotations.iloc[index]["id"] + ".jpg")
        label_path = os.path.join(self.labels_dir, self.annotations.iloc[index]["id"] + ".txt")

        # Get image and bbox within the file
        im = Image.open(image_path).resize((self.image_size, self.image_size))
        image = np.array(ImageOps.exif_transpose(im).convert("RGB"))
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

        # Apply augmentations
        if self.transform is not None:
            print("Transforming ...")
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # Create 3 empty targets, one for every prediction scale
        # Every target has the format [3, S, S, 6], ->  6 = [p_object, x, y, w, h, obj_class]
        # I have three scales [13, 26, 52] and for each scale I hace 3 anchor boxes.
        targets:list[torch.Tensor] = [torch.zeros((self.num_anchors // 3, x, x, 6)) for x in self.detection_layers]

        for box in bboxes:

            iou_results = iou(torch.tensor(box[2:4]), self.anchors)

            anchor_indices = iou_results.argsort(descending=True, dim=0)

            x, y, width, height, class_label = box

            has_anchor = [ False for _ in range(len(self.detection_layers)) ]

            for anchor_idx in anchor_indices:

                scale_idx = anchor_idx // self.num_anchors_per_scale
                        
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.detection_layers[scale_idx]

                # Cell coordinates , relative to the image. Coordinates to the (0,0) of the cell.
                i, j = int(S * x), int(S * y) 

                # Analize if there is an object (p0)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:

                    # [p0 <-, box_coordinates, class_label]
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # center of the box relative to the cell
                    x_cell, y_cell = S * x - i, S * y - j  # both between [0,1]

                    # width and height relative to the cell
                    width_cell, height_cell = (width * S, height * S,)  # can be greater than 1 since it's relative to cell

                    # Get box_coordinates
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    # [p0, box_coordinates <-, class_label]
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates

                    # [p0, box_coordinates, class_label <-]
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)

                    # Set this scale to True, since it has an anchor now
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_results[anchor_idx] > self.ignore_iou_thresh:

                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)

def test(debug=False):

    config = cfg()

    # Path to folders
    csv_data = config["paths"]["debug"]["csv_data"]
    images_path = config["paths"]["debug"]["images"]
    labels_path = config["paths"]["debug"]["labels"]

    # Predefined scaled anchors
    anchors = config["anchors"]

    # Detection layers
    detection_layers = config["detection_layers"]

    transform = test_transforms

    dataset = CustomDataset(
        csv_file=csv_data,
        img_dir=images_path,
        label_dir=labels_path,
        image_size=config["image_size"],
        detection_layers=detection_layers,
        anchors=anchors,
        transform=transform,
    )

    anchors = torch.tensor(anchors) / (
        1 / torch.tensor(detection_layers).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for image, target in loader:

        boxes = []

        for i in range(target[0].shape[1]):

            anchor = anchors[i]

            boxes += cells_to_bboxes(target[i], is_preds=False, S=target[i].shape[2], anchors=anchor)[0]

        boxes = nms(boxes, iou_threshold=0.8, threshold=0.7, box_format="midpoint")

        plot_image(image[0].permute(1, 2, 0), boxes)

        if debug:
            obj = target[0][..., 0] == 1
            with open('tensor_values.txt', 'w+') as file:
                torch.set_printoptions(threshold=torch.inf)
                file.write(str(target))


if __name__ == "__main__":
    test()







        

    
        

