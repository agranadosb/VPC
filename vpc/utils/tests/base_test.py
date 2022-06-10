from unittest import TestCase

import torch


class TestBaseUtils(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.batch_size = 10
        self.scale = 13
        self.anchors = 3
        self.classes = 10
        self.threshold = 0.4
        self.boxes = torch.zeros(
            (
                self.batch_size,
                self.scale,
                self.scale,
                self.anchors,
                1 + 4 + self.classes,
            )
        )
