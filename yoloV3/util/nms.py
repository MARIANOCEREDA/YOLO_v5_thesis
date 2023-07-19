import torch
import torchvision.ops as ops
from util.iou import intersection_over_union

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    scores = [ s[1] for s in bboxes ]

    bboxes = [box[2:] for box in bboxes if box[1] > threshold]

    # Conver to other format
    x = [ box[0] for box in bboxes]
    y = [ box[1] for box in bboxes]
    w = [ box[2] for box in bboxes]
    h = [ box[3] for box in bboxes]

    x1 = [(xi - wi / 2) for xi, wi in zip(x, w)]
    y1 = [(yi - hi / 2) for yi, hi in zip(y, h)]
    x2 = [(xi + wi / 2) for xi, wi in zip(x, w)]
    y2 = [(yi + hi / 2) for yi, hi in zip(y, h)]

    converted_bboxes = [ [x1,y1,x2,y2] for x1,y1,x2,y2 in zip(x1,y1,x2,y2)]

    bboxes_after_nms = []

    nms = ops.nms(torch.tensor(converted_bboxes), torch.tensor(scores), iou_threshold)

    return [ bboxes[i] for i in nms]

    '''while bboxes:

        print(bboxes)

        chosen_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or ops.nms()
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms'''