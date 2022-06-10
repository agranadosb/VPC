from typing import Union, Tuple

from torch import Tensor, add, cat
from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    Sequential,
    LeakyReLU,
    AvgPool2d,
    Linear,
    Softmax,
    Flatten,
    ModuleList,
    Upsample,
)


class ConvBlock(Module):
    """This class implements a convolutional block. It is composed of a
    convolutional layer, a batch normalization layer, and a ReLU activation
    function.
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int = 3
        Size of the convolutional kernel
    stride : int = 1
        Stride of the convolutional kernel
    padding : int = 1
        Padding of the convolutional kernel
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[str, int] = 1,
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = BatchNorm2d(out_channels)
        self.relu = LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(Module):
    """This class implements a residual block.
    Parameters
    ----------
    channels : int
        Number of channels in the input and output tensors.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1, 1, 0)
        self.conv2 = ConvBlock(channels // 2, channels, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))
        return add(out, x)


class DarknetBlock(Module):
    """This class implements a darknet block.

    Parameters
    ----------
    channels : int
        Number of channels in the input and output tensors.
    """

    def __init__(self, channels: int, repetitions: int):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.layers = Sequential(*[ResidualBlock(channels) for _ in range(repetitions)])

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Darknet53(Module):
    """Model for darknet that will be used on the yolo network.

    Parameters
    ----------
    input_size : int = 256
        Size of the input image.
    num_classes : int = 20
        Number of classes to be predicted.
    tail : bool = True
        Whether to use the tail or not.
    """

    def __init__(
        self, input_size: int = 256, num_classes: int = 20, tail: bool = False
    ) -> None:
        super(Darknet53, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        layers = [
            ConvBlock(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            DarknetBlock(channels=64, repetitions=1),
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            DarknetBlock(channels=128, repetitions=2),
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            DarknetBlock(channels=256, repetitions=8),
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            DarknetBlock(channels=512, repetitions=8),
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1
            ),
            DarknetBlock(channels=1024, repetitions=4),
        ]
        if tail:
            dense_size = 1024 * (input_size // 2**6) ** 2
            layers += [
                AvgPool2d(kernel_size=2, stride=2),
                Flatten(),
                Linear(in_features=dense_size, out_features=num_classes),
                Softmax(dim=1),
            ]
        self.layers = ModuleList()
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class YoloUpSample(Module):
    """This class implements the upsampling layer for the yolo network.

    Parameters
    ----------
    channels : int
        Number of channels in the input tensor.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.layers = Sequential(
            ConvBlock(channels, channels // 2), Upsample(scale_factor=2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class YoloPrediction(Module):
    """Yolo prediction layer.

    Parameters
    ----------
    channels : int
        Number of channels in the input tensor.
    num_classes : int = 20
        Number of classes to be predicted.
    num_anchors : int = 3
        Number of anchors to be predicted.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int = 20,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_features_per_anchor = 1 + 4 + num_classes
        self.features = self.num_anchors * self.num_features_per_anchor

        self.layers = Sequential(
            ConvBlock(channels, channels * 2),
            Conv2d(channels * 2, self.features, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        batch_size, channels, height, width = x.shape
        x = x.view(
            batch_size, self.num_anchors, self.num_features_per_anchor, height, width
        )
        return x.permute(0, 3, 4, 1, 2)


class YoloConvPrediction(Module):
    """Yolo convolution prediction layer.

    Parameters
    ----------
    channels : int
        Number of channels in the input tensor.
    """

    def __init__(self, channels: int):
        super().__init__()
        in_channels = channels
        out_channels = channels // 2
        layers = []
        for _ in range(5):
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels, out_channels = out_channels, in_channels
        self.in_channels = out_channels
        self.out_channels = in_channels
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class YoloV3(Darknet53):
    """Yolo v3 model.

    Parameters
    ----------
    input_size : int = 416
        Input size of the model.
    num_classes : int = 20
        Number of classes to be predicted.
    """

    def __init__(self, input_size: int = 416, num_classes: int = 20) -> None:
        super().__init__(input_size=input_size, num_classes=num_classes, tail=False)
        self.prediction1_darknet = self.layers[-1]
        self.prediction2_darknet = self.layers[-3]
        self.prediction3_darknet = self.layers[-5]

        self.prediction1_conv = YoloConvPrediction(
            self.prediction1_darknet.out_channels
        )
        self.prediction2_conv = YoloConvPrediction(
            self.prediction2_darknet.out_channels
            + self.prediction1_conv.out_channels // 2
        )
        self.prediction3_conv = YoloConvPrediction(
            self.prediction3_darknet.out_channels
            + self.prediction2_conv.out_channels // 2
        )

        self.prediction2_upsample = YoloUpSample(self.prediction1_conv.out_channels)
        self.prediction3_upsample = YoloUpSample(self.prediction2_conv.out_channels)

        self.prediction1 = YoloPrediction(self.prediction1_conv.out_channels, num_classes=self.num_classes)
        self.prediction2 = YoloPrediction(self.prediction2_conv.out_channels, num_classes=self.num_classes)
        self.prediction3 = YoloPrediction(self.prediction3_conv.out_channels, num_classes=self.num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        for layer in self.layers[:-5]:  # noqa
            x = layer(x)

        x3 = self.layers[-5](x)
        x_aux = self.layers[-4](x3)
        x2 = self.layers[-3](x_aux)
        x_aux = self.layers[-2](x2)
        x1 = self.layers[-1](x_aux)

        # Prediction 1
        x1 = self.prediction1_conv(x1)
        out1 = self.prediction1(x1)

        # Prediction 2
        concat_1 = self.prediction2_upsample(x1)
        x2 = cat([concat_1, x2], dim=1)
        x2 = self.prediction2_conv(x2)
        out2 = self.prediction2(x2)

        # Prediction 3
        concat_2 = self.prediction3_upsample(x2)
        x3 = cat([concat_2, x3], dim=1)
        x3 = self.prediction3_conv(x3)
        out3 = self.prediction3(x3)

        return out1, out2, out3
