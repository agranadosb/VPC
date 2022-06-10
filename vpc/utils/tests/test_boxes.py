import torch

from vpc.utils.boxes import to_box, box_area, intersection_over_union, to_yolo
from vpc.utils.tests.base_test import TestBaseUtils


class TestUtilsBoxes(TestBaseUtils):
    def test_to_box(self):
        boxes = self.boxes.clone()
        boxes[0, 0, 0, 0, :5] = torch.Tensor([0.5, 0.7, 0.7, 7.0, 7.0])
        correct_boxes = [
            torch.zeros(
                (
                    self.boxes.shape[0],
                    self.boxes.shape[1],
                    self.boxes.shape[2],
                    self.boxes.shape[3],
                )
            )
            for _ in range(4)
        ]
        correct_boxes[0][0, 0, 0, 0] = 0.0
        correct_boxes[1][0, 0, 0, 0] = 0.0
        correct_boxes[2][0, 0, 0, 0] = 4.2
        correct_boxes[3][0, 0, 0, 0] = 4.2

        results = to_box(boxes[..., 1:5])

        for index, correct_box in enumerate(correct_boxes):
            self.assertEqual(results[index].shape, correct_box.shape)
            self.assertTrue(torch.all(results[index].eq(correct_box)))

    def test_to_box_area(self):
        correct_area = torch.zeros((self.batch_size, self.scale, self.scale, self.anchors))
        correct_area[0, 0, 0, 0] = 17.64
        boxes = [
            torch.zeros(
                (
                    self.boxes.shape[0],
                    self.boxes.shape[1],
                    self.boxes.shape[2],
                    self.boxes.shape[3],
                )
            )
            for _ in range(4)
        ]
        boxes[0][0, 0, 0, 0] = 0.0
        boxes[1][0, 0, 0, 0] = 0.0
        boxes[2][0, 0, 0, 0] = 4.2
        boxes[3][0, 0, 0, 0] = 4.2

        result = box_area(*boxes)

        self.assertEqual(result.shape, correct_area.shape)
        self.assertTrue(result.allclose(correct_area))

    def test_intersection_over_union_correct(self):
        boxes = self.boxes.clone()
        labels = self.boxes.clone()
        boxes[0, 0, 0, 0, :] = torch.Tensor(
            [0.5, 0.7, 0.7, 7.0, 7.0] + [0.0] * self.classes
        )
        labels[0, 0, 0, 0, :] = torch.Tensor(
            [0.5, 0.5, 0.5, 5.0, 5.0] + [0.0] * self.classes
        )
        area_box_1 = 4.2 * 4.2
        area_box_2 = 3. * 3.
        intersection = 3. * 3.
        iou = intersection / (area_box_1 + area_box_2 - intersection)
        correct_iou = torch.zeros((
            self.boxes.shape[0],
            self.boxes.shape[1],
            self.boxes.shape[2],
            self.boxes.shape[3],
        ))
        correct_iou[0, 0, 0, 0] = iou

        result = intersection_over_union(boxes, labels)

        self.assertEqual(result.shape, correct_iou.shape)
        self.assertTrue(result.allclose(correct_iou))

    def test_to_yolo(self):
        size = 416, 416
        coordinates = 1, 1, 2, 2
        correct_box = 3 / 832, 3 / 832, 1 / 416, 1 / 416

        result = to_yolo(size, coordinates)

        self.assertTupleEqual(result, correct_box)
