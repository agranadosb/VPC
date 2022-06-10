from typing import Tuple

import torch
from torch import Tensor


def to_yolo(
    size: Tuple[int, int], box: Tuple[int, int, int, int]
) -> Tuple[float, float, float, float]:
    """Convert a bounding box from xmin, ymin, xmax, ymax format to yolo
    format (x, y, width, height).

    Parameters
    ----------
    size: Tuple[int, int]
        Size of the image.
    box: Tuple[int, int, int, int]
        Bounding box in voc format.

    Returns
    -------
    Tuple[float, float, float, float] : Bounding box in yolo format.
    """
    dw = 1 / (size[0])
    dh = 1 / (size[1])
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def iou_width_height(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute the IoU of one bounding box from width and height and a tensor of anchors.

    Parameters
    ----------
    boxes1 Tensor:
        Width and height of the first bounding boxes
    boxes2 Tensor:
        Width and height of the second bounding boxes

    Returns
    -------
    Tensor : Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def to_box(boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert a tensor of bounding boxes in YOLO format
    (cell x, cell y, anchor relative width, anchor relative height) to
    (xmin, ymin, xmax, ymax) format.

    For example, if we have the result of yolo format for one scale:
    ```python
    batch_size = 1
    scale = 13
    anchors = 3
    classes = 10

    boxes = torch.zeros(
        (batch_size, scale, scale, anchors, 1 + 4 + classes)
    )
    boxes[0, 0, 0, 0, :5] = torch.Tensor([0.5, 0.7, 0.7, 7.0, 7.0])
    ```

    The result is:

    ```python
    xmin, ymin, xmax, ymax = to_box(boxes[..., 1:5])
    xmin[0, 0, 0, 0] == 0.0
    ymin[0, 0, 0, 0] == 0.0
    xmax[0, 0, 0, 0] == 4.2
    ymax[0, 0, 0, 0] == 4.2
    ```

    Parameters
    ----------
    boxes Tensor:
        Bounding boxes in YOLO format

    Returns
    -------
    Tuple : Tuple of xmin, ymin, xmax, ymax of the bounding boxes.
    """
    # TODO: Correct boxes based on image dimensions
    xmin = boxes[..., 0] - boxes[..., 2] / 2
    ymin = boxes[..., 1] - boxes[..., 3] / 2
    xmax = boxes[..., 0] + boxes[..., 2] / 2
    ymax = boxes[..., 1] + boxes[..., 3] / 2

    return xmin.clamp(0), ymin.clamp(0), xmax.clamp(0), ymax.clamp(0)


def box_area(xmin: Tensor, ymin: Tensor, xmax: Tensor, ymax: Tensor) -> Tensor:
    """Compute the area of a Tensors of coordinates of bounding boxes.

    Parameters
    ----------
    xmin Tensor:
        xmin of the bounding boxes
    ymin Tensor:
        ymin of the bounding boxes
    xmax Tensor:
        xmax of the bounding boxes
    ymax Tensor:
        ymax of the bounding boxes

    Returns
    -------
    Tensor : Area of the bounding boxes
    """
    return (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0)


def intersection_over_union(predictions: Tensor, labels: Tensor) -> Tensor:
    """This function calculates intersection over union (iou) given prediction
    boxes of Yolo and ground truth boxes.

    Parameters
    ----------
    predictions Tensor:
        Predictions of Bounding Boxes
    labels Tensor:
        Correct labels of Bounding Boxes

    Returns
    -------
    Tensor: Intersection over union for all examples
    """
    (
        xmin_prediction,
        ymin_prediction,
        xmax_prediction,
        ymax_prediction,
    ) = to_box(predictions[..., 1:5])
    xmin_labels, ymin_labels, xmax_labels, ymax_labels = to_box(labels[..., 1:5])

    xmin_intersection = torch.max(xmin_prediction, xmin_labels)
    ymin_intersection = torch.max(ymin_prediction, ymin_labels)
    xmax_intersection = torch.min(xmax_prediction, xmax_labels)
    ymax_intersection = torch.min(ymax_prediction, ymax_labels)

    intersection = box_area(
        xmin_intersection, ymin_intersection, xmax_intersection, ymax_intersection
    )
    box1_area = box_area(
        xmin_prediction, ymin_prediction, xmax_prediction, ymax_prediction
    )
    box2_area = box_area(xmin_labels, ymin_labels, xmax_labels, ymax_labels)

    iou = intersection / (box1_area + box2_area - intersection)
    iou[torch.isnan(iou)] = 0.0
    return iou
