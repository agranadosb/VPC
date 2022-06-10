from unittest import TestCase

import numpy as np
import torch

from vpc.utils.data import (
    serialize_object,
    serialize,
    data_to_list,
    normalize_transformation,
    to_prediction,
)


class TestUtilsData(TestCase):
    def test_serialize_object_no_size(self):
        object_to_serialize = {
            "bndbox": {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2},
            "name": "test",
        }
        serialized_object = 1, 1, 2, 2, "test"

        result = serialize_object(object_to_serialize)

        self.assertTupleEqual(result, serialized_object)

    def test_serialize_object_with_size(self):
        object_to_serialize = {
            "bndbox": {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2},
            "name": "test",
        }
        size = 416, 416
        serialized_object = 3 / 832, 3 / 832, 1 / 416, 1 / 416, "test"

        result = serialize_object(object_to_serialize, size)

        self.assertTupleEqual(result, serialized_object)

    def test_serialize(self):
        size = 416
        image = [[[255, 255, 255]] * (size + 100) for _ in range((size + 100))]
        data_to_serialize = (
            image,
            {
                "annotation": {
                    "filename": "file",
                    "size": {
                        "width": size,
                        "height": size,
                    },
                    "object": [
                        {
                            "bndbox": {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2},
                            "name": "test",
                        }
                    ],
                }
            },
        )
        correct_serialization = (
            np.array(image),
            {
                "name": "file",
                "object": [(1, 1, 2, 2, "test")],
                "yolo": [(3 / 832, 3 / 832, 1 / 416, 1 / 416, "test")],
            },
        )

        result = serialize(data_to_serialize)

        self.assertTrue(np.array_equal(result[0], correct_serialization[0]))
        self.assertTrue(type(result[0]) is np.ndarray)
        self.assertDictEqual(result[1], correct_serialization[1])

    def test_data_to_list(self):
        data = {
            "name": "file",
            "object": [(1, 1, 2, 2, "test")],
            "yolo": [(3 / 832, 3 / 832, 1 / 416, 1 / 416, "test")],
        }
        correct_data = [(1, 1, 2, 2)], ["test"]

        result = data_to_list(data)

        self.assertTupleEqual(result, correct_data)

    def test_normalize_transformation(self):
        size = 416
        image = np.array(
            [[[255.0, 255.0, 255.0]] * (size + 100) for _ in range((size + 100))]
        )
        correct_image = torch.Tensor([[[1.0, 1.0, 1.0]] * size for _ in range(size)])

        transformation = normalize_transformation(size)
        result = transformation(image=image, bboxes=[])

        self.assertTrue(torch.allclose(result["image"], correct_image.permute(2, 0, 1)))

    def test_to_prediction(self):
        size = 416
        scales = [13] * 3
        anchors = torch.Tensor([(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)] * 3)
        classes = ["0"]
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
        correct_box = torch.Tensor([
            1.,
            75 / size * 13 - 2,
            75 / size * 13 - 2,
            50 / size * 13,
            50 / size * 13,
            0.
        ])
        correct_prediction_scale_1 = torch.zeros((13, 13, 3, 1 + 4 + 1))
        correct_prediction_scale_2 = correct_prediction_scale_1.clone()
        correct_prediction_scale_3 = correct_prediction_scale_1.clone()
        correct_prediction_scale_1[2, 2, 0, :] = correct_box
        correct_prediction_scale_2[2, 2, 0, :] = correct_box
        correct_prediction_scale_3[2, 2, 0, :] = correct_box

        image, results = to_prediction(
            image,
            image_data,
            scales=scales,
            anchors=anchors,
            classes=classes,
        )


        self.assertTrue(torch.allclose(results[0], correct_prediction_scale_1))
        self.assertTrue(torch.allclose(results[1], correct_prediction_scale_2))
        self.assertTrue(torch.allclose(results[2], correct_prediction_scale_3))
