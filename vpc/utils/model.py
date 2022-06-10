from typing import Tuple, List, Union

import torch
from torch import Tensor
from torchvision.ops import nms

from vpc.conf.env import SCORE_THRESHOLD
from vpc.constants import SCALES, CLASSES
from vpc.utils.boxes import to_box


def prediction_to_boxes(predictions: Tensor, anchors: Tensor, scale: int) -> Tensor:
    """Converts predictions in yolo format (without applying any operation to
    network output) to bounding boxes (x, y, width, height) and returns the
    class with bigger score. This function use the next formula:

    ```python
    score = sigmoid(to)
    x = (sigmoid(tx) + column) / scale
    y = (sigmoid(ty) + row) / scale
    w = (anchor_width * exp(tw)) / scale
    h = (anchor_height * exp(th)) / scale
    ```

    Being tx, ty, tw, th the predictions of the network for the boxes of each
    cell of each anchor.

    Parameters
    ----------
    predictions : Tensor
        The predictions' tensor with shape:
        ```python
        (batch_size, scale, scale, anchors, 1 + 4 + classes).
        ```
    anchors : Tensor
        The anchors' tensor linked with the predictions' tensor.
    scale : int
        The current scale of the predictions.

    Returns
    -------
    Tensor : The bounding boxes' tensor with linked class.
    """
    assert len(predictions.shape) == 1 + 2 + 1 + 1
    assert predictions.shape == (
        predictions.shape[0],
        scale,
        scale,
        len(anchors),
        predictions.shape[-1],
    )
    anchors_reshaped = anchors.reshape(1, 1, 1, 3, 2)
    indices = (
        torch.Tensor([[index] * 3 for index in range(scale)])
        .repeat(predictions.shape[0], scale, 1, 1)
        .to(predictions.device)
    )
    scale_ratio = 1 / scale

    score = torch.sigmoid(predictions[..., 0]).unsqueeze(-1)
    sizes_boxes = scale_ratio * (torch.exp(predictions[..., 3:5]) * anchors_reshaped)
    best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    coordinates_boxes = torch.sigmoid(predictions[..., 1:3])
    x_boxes = (scale_ratio * (coordinates_boxes[..., 0] + indices)).unsqueeze(-1)
    y_boxes = (
        scale_ratio * (coordinates_boxes[..., 1] + indices.permute(0, 2, 1, 3))
    ).unsqueeze(-1)

    return torch.cat((score, x_boxes, y_boxes, sizes_boxes, best_class), dim=-1)


def to_grounded(
    predictions: Union[Tuple[Tensor, ...], List[Tensor]],
    anchors: Tensor,
    anchors_scale: int = 3,
    dimensions: int = 416,
    *,
    score_threshold: float = SCORE_THRESHOLD,
    nms_threshold: float = 0.4,
    classes: List[str] = CLASSES,
    scales: List[int] = SCALES,
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """Converts predictions of each scale to grounded boxes. Each box is
    composed by xmin, ymin, xmax, ymax and class and these methods returns two
    list, one for boxes and other for classes. The result boxes of the
    predictions will be filtered using non-maximum suppression.

    Parameters
    ----------
    predictions : Tuple[Tensor, Tensor, Tensor]
        The predictions' tensor.
    anchors : Tensor
        The anchors' tensor.
    dimensions : int
        The dimensions of the image.
    anchors_scale : int
        Number of anchors per scale.
    score_threshold : float = `vpc.constants.SCORE_THRESHOLD`
        The threshold to filter the predictions.
    nms_threshold : float = 0.4
        The threshold to apply non-maximum suppression to result boxes.
    classes : List[str] = `vpc.constants.CLASSES`
        The classes labels.
    scales : List[int] = `vpc.constants.SCALES`
        The scales of the predictions.

    Returns
    -------
    Tensor : The grounded boxes' tensor. These boxes are in the form of
    (x, y, width, height, label). The label is the score of the box and the
    class."""
    assert len(scales) * anchors_scale == anchors.shape[0]
    boxes = []
    total_boxes = 0
    for scale in range(len(scales)):
        scale_boxes = prediction_to_boxes(
            predictions[scale],
            anchors[scale * anchors_scale : (scale + 1) * anchors_scale],
            scales[scale],
        )
        boxes.append(scale_boxes[scale_boxes[..., 0] > score_threshold])
        total_boxes += len(boxes[-1])

    if len(boxes) == 0:
        return [], []

    boxes = torch.cat(boxes).reshape(total_boxes, 5 + 1)

    boxes[..., 1:5] = torch.cat(
        [item.unsqueeze(-1) for item in to_box(boxes[..., 1:5])], dim=-1
    )

    grounded_boxes = []
    unique_classes = torch.unique(boxes[..., -1], dim=-1)
    total_boxes = 0
    for box_class in unique_classes:
        class_boxes = boxes[boxes[..., -1] == box_class]
        nms_boxes = nms(class_boxes[..., 1:5], class_boxes[..., 0], nms_threshold)

        total_boxes += len(nms_boxes)
        grounded_boxes.append(class_boxes[nms_boxes])

    if len(grounded_boxes) == 0:
        return [], []

    grounded_boxes = torch.cat(grounded_boxes).reshape(total_boxes, 5 + 1).to("cpu")

    result_boxes = []
    result_labels = []
    for box in grounded_boxes:
        result_boxes.append(
            (
                int(box[1] * dimensions),
                int(box[2] * dimensions),
                int(box[3] * dimensions),
                int(box[4] * dimensions),
            )
        )
        result_labels.append(f"{box[0]:.2f}-{classes[int(box[5])]}")
    return result_boxes, result_labels
