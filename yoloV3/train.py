import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from yoloV3.model import YOLOv3
from config.config import config as cfg
from config.config import architecture_config
from yoloV3.dataset import CustomDataset
from yoloV3.loss import YoloV3Loss
from util.test_transform import test_transforms
from util.mAP import mean_average_precision
from util.check_class_accuracy import check_class_accuracy
from util.get_evaluation_boxes import get_evaluation_bboxes


config = cfg()

def get_dataset_loader(mode:str):

    csv_data = config["paths"][mode]["csv_data"]
    images_path = config["paths"][mode]["images"]
    labels_path = config["paths"][mode]["labels"]

    # Predefined scaled anchors
    anchors = config["anchors"]
    batch_size = config["train"]["batch_size"]

    # Detection layers
    detection_layers = config["detection_layers"]

    dataset = CustomDataset(
        csv_file=csv_data,
        img_dir=images_path,
        label_dir=labels_path,
        image_size=config["image_size"],
        detection_layers=detection_layers,
        anchors=anchors,
        transform=None,
    )

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size, 
                        shuffle=True)

    return loader

def scale_anchors(anchors , detection_layers):

    return (
        torch.tensor(anchors)
        * torch.tensor(detection_layers).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

def train():

    model_arch = architecture_config()

    yolov3_model = YOLOv3(in_channels=3, num_classes=config["num_classes"], arch_config=model_arch).to(config["device"])

    optimizer = optim.Adam(yolov3_model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["wd"])

    scaler = torch.cuda.amp.GradScaler()

    train_loader = get_dataset_loader(mode="train")
    test_loader =  get_dataset_loader(mode="test")

    yolo_loss = YoloV3Loss()

    progress_barr = tqdm(train_loader, leave=True)

    anchors = scale_anchors(config["anchors"], config["detection_layers"]).to(config["device"])

    for epoch in range(config["train"]["epochs"]):

        print(f"Epoch: {epoch}")

        losses = []

        for batch_idx, (image, target) in enumerate(progress_barr):

            image = image.to(config["device"])

            y0, y1, y2 = (
                target[0].to(config["device"]),
                target[1].to(config["device"]),
                target[2].to(config["device"]),
            )

            with torch.cuda.amp.autocast():

                image = image.type(torch.cuda.HalfTensor)
                out = yolov3_model(image.permute(0,3,1,2))

                total_loss = (
                    yolo_loss(out[0], y0, anchors[0])
                    + yolo_loss(out[1], y1, anchors[1])
                    + yolo_loss(out[2], y2, anchors[2])
                )
            
            losses.append(total_loss.item())
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            progress_barr.set_postfix(loss=mean_loss)
        
        if epoch > 0 and epoch % 3 == 0:

            check_class_accuracy(yolov3_model, test_loader, threshold=config["train"]["threshold"], device=config["device"])

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                yolov3_model,
                iou_threshold=config["train"]["nms_threshold"],
                anchors=config["anchors"],
                threshold=config["train"]["threshold"],
            )

            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config["train"]["iou_threshold"],
                box_format="midpoint",
                num_classes=config["num_classes"],
            )

            print(f"MAP: {mapval.item()}")

            yolov3_model.train()

if __name__ == "__main__":
    train()