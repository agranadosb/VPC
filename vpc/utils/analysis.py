from typing import List, Tuple, Union

import cv2 as cv
import matplotlib.colors as mat_colors
import numpy as np
import torchvision.transforms as T  # noqa
from torch import Tensor

from vpc.constants.types import TYPE_DATA
from vpc.constants import COLORS
from vpc.utils.data import data_to_list
from vpc.utils.model import to_grounded
from vpc.utils.text import text_ellipsis


def show_bounding_boxes(
    image: np.ndarray, boxes: List[Tuple[int, int, int, int]], classes: List[str]
) -> np.ndarray:
    """Visualizes prediction classes, bounding boxes over the source image
    and exports it to output folder.

    Parameters
    ----------
    image: np.ndarray
        Source image.
    boxes: list[tuple[int, int, int, int]]
        List of bounding boxes.
    classes: list[str]
        List of classes."""
    assert len(boxes) == len(classes)

    for i, (box, label) in enumerate(zip(boxes, classes)):
        color = mat_colors.hex2color(COLORS[i % len(COLORS)])
        color = color[0] * 255, color[1] * 255, color[2] * 255

        bottom_left_corner = box[:2]
        top_right_corner = box[2:]

        text = text_ellipsis(label, max_size=20)
        text_size = (top_right_corner[0] - bottom_left_corner[0]) / 150 * 0.5
        if text_size < 0.5:
            text_size = 0.5

        width, height = cv.getTextSize(text, 0, fontScale=text_size, thickness=5)[0]

        top_right_corner_box_label = [
            bottom_left_corner[0] + width,
            bottom_left_corner[1] - height - 3,
        ]
        bottom_left_corner_box_label = [
            bottom_left_corner[0] - 2,
            bottom_left_corner[1],
        ]
        if top_right_corner_box_label[0] < 0 or bottom_left_corner_box_label[0] < 0:
            value = (
                min(top_right_corner_box_label[0], bottom_left_corner_box_label[0]) * -1
            )
            top_right_corner_box_label[0] += value + 10
            bottom_left_corner_box_label[0] += value + 10
        if top_right_corner_box_label[1] < 0 or bottom_left_corner_box_label[1] < 0:
            value = (
                min(top_right_corner_box_label[1], bottom_left_corner_box_label[1]) * -1
            )
            top_right_corner_box_label[1] += value + 10
            bottom_left_corner_box_label[1] += value + 10

        x_text = bottom_left_corner_box_label[0] - 2
        y_text = bottom_left_corner_box_label[1] - 3

        cv.rectangle(
            image,
            bottom_left_corner,
            top_right_corner,
            color=color,
            thickness=5,
        )
        cv.rectangle(
            image,
            bottom_left_corner_box_label,
            top_right_corner_box_label,
            color,
            -1,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            text,
            (x_text, y_text),
            0,
            text_size,
            (255, 255, 255),
            thickness=1,
        )
    return image


def show_image_data(image: np.ndarray, data: TYPE_DATA) -> np.ndarray:
    """Shows an image and its data.

    Parameters
    ----------
    image: np.ndarray
        Image to be shown.
    data: dict[str, Any]
        Data to be shown."""
    return show_bounding_boxes(image, *data_to_list(data))


def show_prediction(
    image: Tensor, predictions: Tuple[Tensor, Tensor, Tensor], anchors: Tensor
) -> Union[np.ndarray, None]:
    """Shows an image and its prediction.

    Parameters
    ----------
    image: torch.Tensor
        Image to be shown.
    predictions: Tuple[Tensor, Tensor, Tensor]
        Prediction to be shown.
    anchors: torch.Tensor
        Anchors to be used."""
    boxes, labels = to_grounded(predictions, anchors)
    if len(boxes) == 0:
        return None
    image = np.asarray(T.ToPILImage()(image))
    return show_bounding_boxes(image, boxes, labels)
