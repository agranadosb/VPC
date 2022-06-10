import numpy as np
import torch

from vpc.utils.model import prediction_to_boxes, to_grounded
from vpc.utils.tests.base_test import TestBaseUtils


class TestUtilsModel(TestBaseUtils):
    def setUp(self) -> None:
        super().setUp()
        self.anchors = torch.Tensor([(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)] * 3)

    def test_from_prediction_to_boxes(self):
        boxes = self.boxes.clone()
        boxes[0, 0, 0, 0, :] = torch.Tensor(
            [0.28, 0.22, 0.38, 0.48, 0.9] + [0.7] + [0.0] * (self.classes - 1)
        )
        correct = torch.Tensor(
            [
                np.exp(-np.logaddexp(0, -0.28)),
                np.exp(-np.logaddexp(0, -0.22)) / self.scale,
                np.exp(-np.logaddexp(0, -0.38)) / self.scale,
                np.exp(0.48) * 0.28 / self.scale,
                np.exp(0.9) * 0.22 / self.scale,
                0.0,
            ]
        )

        result = prediction_to_boxes(boxes, self.anchors[:3], self.scale)

        self.assertEqual(result[0, 0, 0, 0, :].shape, correct.shape)
        self.assertTrue(torch.allclose(result[0, 0, 0, 0, :], correct))

    def test_to_grounded(self):
        boxes = self.boxes.clone()
        boxes[0, 0, 0, 0, :] = torch.Tensor(
            [0.28, 0.22, 0.38, 0.48, 0.9] + [0.7] + [0.0] * (self.classes - 1)
        )
        x = np.exp(-np.logaddexp(0, -0.22)) / self.scale
        y = np.exp(-np.logaddexp(0, -0.38)) / self.scale
        w = np.exp(0.48) * 0.28 / self.scale
        h = np.exp(0.9) * 0.22 / self.scale
        correct_boxes = [
            (
                int((x - w / 2) * 416),
                int((y - h / 2) * 416),
                int((x + w / 2) * 416),
                int((y + h / 2) * 416),
            )
        ]
        correct_labels = [f"{np.exp(-np.logaddexp(0, -0.28)):.2f}-0"]

        result_boxes, result_labels = to_grounded(
            [boxes for _ in range(3)],
            self.anchors,
            score_threshold=0.55,
            nms_threshold=0.55,
            classes=[str(i) for i in range(10)],
            scales=[13] * 3,
        )

        self.assertEqual(len(result_boxes), 1)
        self.assertEqual(len(correct_labels), 1)
        self.assertListEqual(correct_boxes, result_boxes)
        self.assertListEqual(correct_labels, result_labels)
