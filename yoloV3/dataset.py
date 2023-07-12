from ..config.config import config as cfg
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

config = cfg()

class CustomDataset(Dataset):

    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        detection_layer_size=[13, 26, 52],
        num_classes=1,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.detection_layer_size = detection_layer_size
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> any:
        
        

    
        

