import torch
from torch import Tensor
from torch import nn

from vpc.utils.boxes import intersection_over_union


class YoloV3Loss(nn.Module):
    """
    YOLOv3 Loss function.

    Parameters
    ----------
    lambda_coord : int
        Weight of bounding box coordinates.
    lambda_noobj : int
        Weight of no-object loss.
    """

    def __init__(self, lambda_coord: int = 10, lambda_noobj: int = 10):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()

        self.sigmoid = nn.Sigmoid()

    def forward(self, pred: Tensor, target: Tensor, anchors: Tensor) -> Tensor:
        # prediction -> Batch size x (grid_size x grid_size x num_anchors) x (5 + num_classes)
        # Get cells that have objects and cells that don't have objects
        cells_with_objects = target[..., 0] == 1
        cells_without_objects = target[..., 0] == 0

        # BOXES LOSS
        anchors_reshaped = anchors.reshape(1, 1, 1, 3, 2)
        sizes_boxes = torch.exp(pred[..., 3:5]) * anchors_reshaped
        coordinates_boxes = self.sigmoid(pred[..., 1:3])
        boxes_data = torch.cat((coordinates_boxes, sizes_boxes), dim=-1)
        box_loss = self.mse(
            boxes_data[cells_with_objects], target[..., 1:5][cells_with_objects]
        )
        boxes_data = torch.cat((pred[..., 0].unsqueeze(-1), boxes_data), dim=-1)

        # CONFIDENCE LOSS
        # iou_boxes = intersection_over_union(
        #     boxes_data[cells_with_objects], target[..., 0:5][cells_with_objects]
        # )
        object_confidence_loss = self.bce(
            self.sigmoid(pred[..., 0][cells_with_objects]),
            target[..., 0][cells_with_objects],
        )
        no_object_confidence_loss = self.bce(
            self.sigmoid(pred[..., 0][cells_without_objects]), target[..., 0][cells_without_objects]
        )

        # CLASS LOSS
        class_loss = self.entropy(
            (pred[..., 5:][cells_with_objects]),
            (target[..., 5][cells_with_objects].long()),
        )

        return (
            self.lambda_noobj * no_object_confidence_loss
            + object_confidence_loss
            + self.lambda_coord * box_loss
            + class_loss
        )
