import datetime
import logging
import math
import os.path
from typing import List, Tuple

import torch
import tqdm as tqdm
from torch import manual_seed, Tensor, device as torch_device, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from vpc.conf.env import EPOCHS, STEPS_FOR_APPLY_GRADIENT, LOG_DIR
from vpc.constants import CLASSES, SCALES, ANCHORS
from vpc.data import VOCDataset, load_datasets
from vpc.model import YoloV3
from vpc.model.loss import YoloV3Loss
from vpc.utils.analysis import show_prediction
from vpc.utils.data import normalize_transformation

DEVICE = torch_device("cuda" if cuda.is_available() else "cpu")

manual_seed(17)


class VocTrainerYoloV3:
    """
    Trainer for YoloV3 model.

    Parameters
    ----------
    dimensions: int
        Number of dimensions of the input image.
    batch_size: int
        Number of samples in a batch.
    epochs: int
        Number of epochs to train the model.
    learning_rate: float
        Learning rate for the optimizer.
    gradient_steps: int
        Number of steps to apply gradient.
    log_dir: str
        Path to the directory where the logs will be saved.
    """

    def __init__(
        self,
        dimensions: int,
        batch_size: int,
        epochs: int,
        data_folder: str,
        anchors: Tensor = ANCHORS,
        anchors_scale: int = 3,
        scales: List[int] = SCALES,
        learning_rate: float = 0.0001,
        training_ratio: float = 0.8,
        gradient_steps: int = None,
        log_dir: str = None,
    ):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_folder = data_folder
        self.learning_rate = learning_rate
        self.training_ratio = training_ratio

        self.anchors = torch.cat(
            [
                anchors[index * anchors_scale : (index + 1) * anchors_scale]
                for index in range(len(scales))
            ],
            dim=-2,
        )

        self.log_dir = log_dir
        if log_dir is None:
            self.log_dir = LOG_DIR

        base_path, folder = os.path.split(self.log_dir)
        folder = f"{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}-{folder}"
        if not base_path:
            base_path = os.getcwd()
        self.log_dir = os.path.join(base_path, folder)
        logging.info("Data saved on: ", self.log_dir)

        self.gradient_steps = gradient_steps
        if gradient_steps is None:
            self.gradient_steps = STEPS_FOR_APPLY_GRADIENT

        self.num_classes = len(CLASSES)
        self.model = YoloV3(self.dimensions, num_classes=self.num_classes)

        self.training = None
        self.validation = None
        self.loss_function = None
        self.optimizer = None
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.losses = []

    def load_datasets(self) -> "VocTrainerYoloV3":
        """This function loads the datasets for training and validation."""
        self.training, self.validation = load_datasets(
            self.data_folder, self.dimensions, self.batch_size, self.training_ratio
        )
        return self

    def plot_model(self) -> "VocTrainerYoloV3":
        """This function plots the model."""
        summary(
            self.model,
            input_size=(self.batch_size, 3, self.dimensions, self.dimensions),
        )

        training = VOCDataset(
            self.data_folder,
            images_transformations=normalize_transformation(self.dimensions),
        )

        images, _ = next(
            iter(DataLoader(training, shuffle=True, batch_size=self.batch_size))
        )

        images = images.to(DEVICE)
        self.writer.add_graph(self.model, images)

        return self

    def show_predictions(self, images: Tensor, maximum: int = 1) -> List[Tensor]:
        """This function shows the predictions of the model.

        Parameters
        ----------
        images: Tensor
            Images to show predictions.
        maximum: int
            Maximum number of predictions to show.
        """
        predictions = self.model.to("cpu").forward(images)
        anchors = self.anchors.to("cpu")

        images_iterable = zip(images, *predictions)
        bounding_images = []
        for index, (image, prediction_1, prediction_2, prediction_3) in enumerate(
            images_iterable
        ):
            if index > maximum:
                break
            prediction_1 = prediction_1.reshape(1, *prediction_1.shape)
            prediction_2 = prediction_2.reshape(1, *prediction_2.shape)
            prediction_3 = prediction_3.reshape(1, *prediction_3.shape)

            result_image = show_prediction(
                image, (prediction_1, prediction_2, prediction_3), anchors
            )
            if result_image is not None:
                bounding_images += [
                    Tensor(image).unsqueeze(0),
                    Tensor(result_image).permute(2, 0, 1).unsqueeze(0),
                ]

        self.model.to(DEVICE)
        self.anchors.to(DEVICE)
        return bounding_images

    def evaluate(self, epoch: int) -> "VocTrainerYoloV3":
        """Gets metrics on validation set.

        Parameters
        ----------
        epoch: int
            Number of epoch.
        """
        losses = []
        bounding_images = []
        show = True
        with torch.no_grad():
            for index, (images, targets) in enumerate(tqdm.tqdm(self.validation)):
                if len(bounding_images) < 10:
                    bounding_images += self.show_predictions(images)
                if len(bounding_images) == 10 and show:
                    self.writer.add_images(
                        f"Evaluation images epoch {epoch}",
                        torch.concat(bounding_images, dim=0),
                    )
                    show = False

                images, targets = self._to_device(images, targets)
                losses.append(self._forward(images, targets).item())

        if len(bounding_images) > 0 and show:
            self.writer.add_images(
                f"Evaluation images epoch {epoch}",
                torch.concat(bounding_images, dim=0),
            )

        mean = math.fsum(losses) / len(losses)
        self.writer.add_scalar(f"val-loss", mean, epoch)

        print(f"Validation loss: {mean}")

        return self

    def _update_analysis(
        self, progress_bar: tqdm.tqdm, epoch: int, batch: int
    ) -> "VocTrainerYoloV3":
        """This function updates the analysis of the training.

        Parameters
        ----------
        progress_bar: tqdm.tqdm
            Progress bar.
        epoch: int
            Current epoch.
        batch: int
            Current batch.
        """
        mean_loss = math.fsum(self.losses) / len(self.losses)
        self.writer.add_scalar("loss", mean_loss, epoch)
        self.writer.add_scalar(f"loss-epoch-{epoch}", self.losses[-1], batch)

        progress_bar.set_description(
            f"Device: {DEVICE} - Epoch {epoch:3} - Loss {mean_loss:16.2f}"
        )
        progress_bar.update(1)

        return self

    def _to_device(  # noqa
        self, images: Tensor, targets: List[Tensor]
    ) -> Tuple[Tensor, List[Tensor]]:  # noqa
        """This function moves the tensors to the device.

        Parameters
        ----------
        images: torch.Tensor
            Images tensor.
        targets: torch.Tensor
            Targets tensor.
        """

        images = images.to(DEVICE)
        targets = [
            targets[0].to(DEVICE),
            targets[1].to(DEVICE),
            targets[2].to(DEVICE),
        ]
        return images, targets

    def _forward(self, images: Tensor, targets: List[Tensor]) -> Tensor:
        """Applies forward pass.

        Parameters
        ----------
        images: Tensor
            Batch of images.
        targets: List[Tensor]
            Batch of targets.

        Returns
        -------
        Tensor : Loss.
        """
        predictions = self.model.forward(images)

        loss_first_scale = self.loss_function.forward(
            predictions[0].float(), targets[0], self.anchors[0:3]
        )
        loss_second_scale = self.loss_function.forward(
            predictions[1].float(), targets[1], self.anchors[3:6]
        )
        loss_third_scale = self.loss_function.forward(
            predictions[2].float(), targets[2], self.anchors[6:]
        )

        return loss_first_scale + loss_second_scale + loss_third_scale

    def _train_step(
        self, images: Tensor, targets: List[Tensor], batch: int
    ) -> "VocTrainerYoloV3":
        """
        This function makes a train step. This function makes a forward pass
        with only 1 batch. If the module of batch index with gradient_steps is 0 a
        gradient descent is applied.

        Parameters
        ----------
        images: Tensor
            Batch of images.
        targets: List[Tensor]
            Batch of targets.
        batch: int
            Batch index.
        """
        total_loss = self._forward(images, targets)
        self.losses.append(total_loss.item())
        total_loss.backward()

        if batch % self.gradient_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return self

    def _epoch_loop(self, epoch: int) -> "VocTrainerYoloV3":
        """This function makes a loop over the batches of the epoch.

        Parameters
        ----------
        epoch: int
            Current epoch.
        """
        self.optimizer.zero_grad()
        with tqdm.tqdm(total=len(self.training)) as progress_bar:
            for batch, (images, targets) in enumerate(self.training):
                images, targets = self._to_device(images, targets)
                self._train_step(images, targets, batch)._update_analysis(
                    progress_bar, epoch, batch
                )
            if batch % self.gradient_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.evaluate(epoch)
        return self

    def train(self) -> "VocTrainerYoloV3":
        """This function trains the model. The optimizer used by this function
        is Adam"""
        self.loss_function = YoloV3Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.load_datasets()
        self.model.to(DEVICE)
        self.anchors = self.anchors.to(DEVICE)
        self.evaluate(0)
        for epoch in range(1, EPOCHS + 1):
            self._epoch_loop(epoch)
        return self
