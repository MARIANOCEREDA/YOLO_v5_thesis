import torch
from torch import nn
from util.iou import intersection_over_union

class YoloV3Loss(nn.Module):

    def __init__(self,  **kwargs) -> None:
        super().__init__( **kwargs)

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.l_class = 1
        self.l_noobj = 10
        self.l_obj = 1
        self.l_box = 10

    def _no_obj_loss(self, predictions:torch.Tensor, noobj:list[bool], target:torch.Tensor):
        """
        No object loss: For cells which have not objects, the target will be all zeros (since there is not bbox).

        For the anchors in all cells that do not have an object assigned to them i.e. all indices that are set to one in "noobj" 
        we want to incur loss only for their object score. The target will be all zeros since we want these anchors to predict 
        an object score of zero and we will apply a sigmoid function to the network outputs and use a binary crossentropy loss
        """
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),)
        
        return no_object_loss
    
    def _object_loss(self, predictions:torch.Tensor, obj:list[bool], target:torch.Tensor, anchors:torch.Tensor):
        """
        Measures the difference between the predicted objectness score (indicating the presence of an object) 
        and the ground truth objectness label (0 for background, 1 for object).

        Parameters:
            - predictions (Tensor) : Predictions of the NN with format [p_o, x, y, w, h, class]
            - obj (Tensor) : Tensor with True value in the pixel that there is an object.
            - target (Tensor) : Target or ground truth values with format [p_o, x, y, w, h, class]
            - anchors (Tensor) : Anchor boxes content.

        Return:
            Tensor : object loss

        """

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # According to the paper bx,by,bx,bh
        b_xy = self.sigmoid(predictions[...,1:3])
        b_wh = torch.exp(predictions[..., 3:5]) * anchors

        # Predicted boxes according to the paper function
        box_preds = torch.cat([b_xy, b_wh], dim=-1)

        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()

        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        return object_loss
    
    def _box_coordinate_loss(self, predictions:torch.Tensor, obj:torch.Tensor, target:torch.Tensor, anchors:torch.Tensor):
        """
        The box coordinate loss is used to train the model's bounding box predictions.
        It calculates the discrepancy between the predicted bounding box coordinates (center x, center y, width, height) 
        and the ground truth box coordinates.

        Parameters:
            - predictions (Tensor) : Predictions of the NN with format [p_o, x, y, w, h, class]
            - obj (Tensor) : Tensor with True value in the pixel that there is an object.
            - target (Tensor) : Target or ground truth values with format [p_o, x, y, w, h, class]
            - anchors (Tensor) : Anchor boxes content.

        Return:
            Tensor : box coordinate loss.

        """

        # Apply a sigmoid function to the x and y coordinates to make sure that they are between [0,1]
        predictions[...,1:3] = self.sigmoid(predictions[..., 1:3])

        print(f"Anchors shape: {anchors.shape}")
        print(f"Target shape: {target[...,3:5].shape}")

        # Compute the ground truth value that the network should predict.
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors)) # convert target width and height

        # Apply finally mean squared error
        box_coord_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        return box_coord_loss

    def _class_predictions(self, predictions:torch.Tensor, obj:torch.Tensor, target:torch.Tensor):
        """
        YOLOv3 can handle multi-class object detection, so the class loss is used to train the model's class predictions.
        For each anchor box, this loss measures the difference between the predicted class probabilities and the ground 
        truth class labels.

        Parameters:
            - predictions (Tensor) : Predictions of the NN with format [p_o, x, y, w, h, class]
            - obj (Tensor) : Tensor with True value in the pixel that there is an object.
            - target (Tensor) : Target or ground truth values with format [p_o, x, y, w, h, class]

        Return:
            Tensor : class loss.

        """
        class_loss = self.entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()),)

        return class_loss

    def _total_loss(self, no_object_loss:torch.Tensor, object_loss:torch.Tensor, box_loss:torch.Tensor, class_loss:torch.Tensor):
        """
        The total loss in YOLOv3 is the sum of the objectness loss, box coordinate loss, and class loss.
        Each loss is weighted based on their respective importance to the overall training objective.
        """

        total_loss = self.l_noobj * no_object_loss + self.l_obj * object_loss 
        + self.l_box * box_loss + self.l_class * class_loss

        return total_loss

    
    def forward(self,  predictions:torch.Tensor, target:torch.Tensor, anchors:torch.Tensor):
        """

        Returns:
            Coordinate loss — due to a box prediction not exactly covering an object,
            Objectness loss — due to a wrong box-object IoU prediction,
            Classification loss — due to deviations from predicting 1for the correct classes and 0
            for all the other classes for the object in that box.
        """

        # Check where obj and noobj (we ignore if target == -1)

        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        no_object_loss = self._no_obj_loss(predictions, noobj, target)

        anchors = anchors.reshape(1, 3, 1, 1, 2)

        object_loss = self._object_loss(predictions, obj, target, anchors)

        box_coord_loss = self._box_coordinate_loss(predictions, obj=obj, target=target, anchors=anchors)

        class_loss = self._class_predictions(predictions=predictions, obj=obj, target=target)

        return self._total_loss(no_object_loss, object_loss, box_coord_loss, class_loss)
        

