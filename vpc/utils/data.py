from typing import Any, Tuple, Dict, List, Iterable, Optional

import albumentations as album
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from vpc.constants import SCALES, ANCHORS, CLASSES
from vpc.constants.types import TYPE_OBJECT, TYPE_DATA, TYPE_PAIR_YOLO
from vpc.utils.boxes import to_yolo, iou_width_height


def serialize_object(
    data: Dict[str, Any], size: Optional[Tuple[int, int]] = None
) -> TYPE_OBJECT:
    """Serialize an object that has a bounding box. The input object must have the next structure:

    ```python
    obj = {
        "name": "name",
        "bbox": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }
    }
    ```

    The result is a tuple in yolo format (x, y, w, h, label).

    Parameters
    ----------
    data: dict[str, Any]
        Object to be serialized.
    size: Tuple[int, int] = None, optional
        Size of the image.

    Returns
    -------
    Serialized object in yolo format (x, y, w, h, label)."""
    bbox = (
        int(data["bndbox"]["xmin"]),
        int(data["bndbox"]["ymin"]),
        int(data["bndbox"]["xmax"]),
        int(data["bndbox"]["ymax"]),
    )
    value = (*bbox, data["name"])
    if size is not None:
        value = (*to_yolo(size, bbox), data["name"])
    return value  # noqa


def serialize(data: Tuple[Iterable, Dict[str, Any]]) -> Tuple[np.ndarray, TYPE_DATA]:
    """Serialize the information of an image. The information is a tuple with
    the image and the information of the image:


    ```python
    data = (
        image,
        {
            "annotation": {
                "filename": "filename",
                "size": {
                    "width": width,
                    "height": height,
                },
                "object": [
                    {
                        "bndbox": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax
                        },
                        "name": "name"
                    }
                ]
            }
        }
    )
    ```

    The result of the serialization is:

    ```python
    result = (
        np.array(image),
        {
            "name": "filename",
            "object": [xmin, ymin, xmax, ymax, "label"],
            "yolo": [x, y, width, height, "label"],
        }
    )
    ```

    Parameters
    ----------
    data: Tuple[Iterable, yolo.constants.TYPE_RAW_DATA]
        Image to be serialized.

    Returns
    -------
    Tuple[np.ndarray, yolo.constants.TYPE_DATA]: Serialized image."""
    image_data = data[1]["annotation"]
    size = int(image_data["size"]["width"]), int(image_data["size"]["height"])

    return (
        np.array(data[0]),
        {
            "name": image_data["filename"],
            "object": [serialize_object(obj) for obj in image_data["object"]],
            "yolo": [serialize_object(obj, size) for obj in image_data["object"]],
        },
    )


def data_to_list(data: TYPE_DATA) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """Converts the data to a list of bounding boxes and a list of labels.

    Parameters
    ----------
    data: dict[str, Any]
        Data to be converted.

    Returns
    -------
    List of bounding boxes and a list of labels."""
    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[str] = []

    objects = data["object"]
    for obj in objects:
        boxes.append(obj[:-1])
        labels.append(obj[-1])

    return boxes, labels


def normalize_transformation(image_dimensions: int) -> album.Compose:
    """Normalize an image.

    Parameters
    ----------
    image_dimensions: int
        Width or height of the image. Both sizes must be equals.

    Returns
    -------
    Sequential : Sequential transformations for normalize an image.
    """
    return album.Compose(
        [
            album.Resize(image_dimensions, image_dimensions),
            album.Normalize(
                mean=[0, 0, 0],
                std=[255, 255, 255],
                max_pixel_value=1,
            ),
            ToTensorV2(),
        ],
        bbox_params=album.BboxParams(
            format="yolo",
            label_fields=[],
        ),
    )


def to_prediction(
    image: Iterable,
    image_data: dict[str, Any],
    *,
    scales: List[int] = SCALES,
    anchors: Tensor = ANCHORS,
    classes: List[str] = CLASSES,
    transforms: Optional[album.Compose] = None,
) -> TYPE_PAIR_YOLO:
    """Transforms an image and its data to a prediction. The image is normalized,
    the bounding boxes are converted to yolo format and the labels are converted
    to index based on classes list parameter. The next code block is an example:

    ```python
    image = np.array([[[255.0, 255.0, 255.0]] * size for _ in range(size)])
    image_data = {
        "annotation": {
            "filename": "file",
            "size": {
                "width": size,
                "height": size,
            },
            "object": [
                {
                    "bndbox": {"xmin": 50, "ymin": 50, "xmax": 100, "ymax": 100},
                    "name": "0",
                }
            ],
        }
    }
    column, row = 2
    correct_box = torch.Tensor([
        1.,
        ((50 + 100) / 2) / size * 13 - 2,
        ((50 + 100) / 2) / size * 13 - 2,
        (100 - 50) / size * 13,
        (100 - 50) / size * 13,
        0.
    ])
    anchors =  torch.Tensor([(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)] * 3)
    scales = [13] * 13

    result = to_prediction(image, image_data, scales=scales, anchors=anchors, classes=["0"])

    result.shape == (13, 13, 3, 1 + 4 + 1)
    result[0][2, 2, 0, :] == correct_box
    result[1][2, 2, 0, :] == correct_box
    result[2][2, 2, 0, :] == correct_box
    ```

    Parameters
    ----------
    image : Iterable
        The image.
    image_data : TYPE_RAW_DATA
        The image data.
    scales : List[int] = vpc.constants.SCALES, optional
        Scales to use.
    anchors : Tensor = vpc.constants.ANCHORS, optional
        Anchors to use.
    classes : List[str] = vpc.constants.CLASSES, optional
        Classes to use.
    transforms : Optional[album.Compose] = None, optional
        Transforms to use.

    Returns
    -------
    Union[TYPE_PAIR, TYPE_PAIR_YOLO] : The serialized image and the image data.
    """
    image, image_data = serialize((image, image_data))

    if transforms is None:
        transforms = normalize_transformation(image.shape[1])
    transformation_result = transforms(image=image, bboxes=image_data["yolo"])
    image: Tensor = transformation_result["image"]
    image_data["yolo"] = transformation_result["bboxes"]

    results: Tuple[Tensor, ...] = tuple([torch.zeros(scale, scale, 3, 1 + 4 + 1) for scale in scales])
    for box_x, box_y, width, height, label in image_data["yolo"]:
        iou_anchors = iou_width_height(torch.Tensor([width, height]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)

        scales_objects = [False] * len(scales)
        for anchor_index in anchor_indices:
            anchor_index = anchor_index.item()

            scale = anchor_index // 3
            anchor = anchor_index % 3

            current_scale = scales[scale]

            column = int(current_scale * box_x)
            row = int(current_scale * box_y)

            result_scale = results[scale]

            has_object = result_scale[row, column, anchor, 0] == 1.0
            if scales_objects[scale] or has_object:
                continue
            scales_objects[scale] = True
            result_scale[row, column, anchor, 0] = 1

            # Sigmoid applied
            cell_x = current_scale * box_x - column
            cell_y = current_scale * box_y - row
            # All steps applied
            cell_width = current_scale * width
            cell_height = current_scale * height

            index_tensor = torch.Tensor([classes.index(label)])

            result_scale[row, column, anchor, 1:5] = torch.Tensor(
                [cell_x, cell_y, cell_width, cell_height]
            )
            result_scale[row, column, anchor, 5] = index_tensor

            if all(scales_objects):
                break

    return image, results
