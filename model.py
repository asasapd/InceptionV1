from typing import Tuple
from torch import nn, relu
import torch
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Dropout2d


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        bias=True,
        normalization=True,
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.normalization = normalization
        self.norm = nn.BatchNorm2d(
            num_features=out_channels, affine=True, eps=0.00001, momentum=0.1
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv2d(x)
        if self.normalization:
            x = self.norm(x)
        x = self.activation(x)
        return x


class Inception_block(nn.Module):
    def __init__(
        self,
        in_channel: int,
        b1_out_channel: Tuple[int, int],
        b2_channels: Tuple[int, int],
        b3_channels: Tuple[int, int],
        b4_channel: int,
    ) -> None:
        """
        b1_out_channel: out channel of 1x1 convolution branch
        b2_channels: out channel of 1x1 and 3x3 conv branch
        b3_channels: out channle of 1x1 and 5x5 conv branch
        b4_channel: out channel of max_pool branch
        """
        super(Inception_block, self).__init__()
        self.branch_1 = BasicConv2d(
            in_channels=in_channel, out_channels=b1_out_channel, kernel_size=1, stride=1
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(
                in_channels=in_channel,
                out_channels=b2_channels[0],
                kernel_size=1,
                stride=1,
                padding=1,
            ),
            BasicConv2d(
                in_channels=b2_channels[0],
                out_channels=b2_channels[1],
                kernel_size=3,
                stride=1,
            ),
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(
                in_channels=in_channel,
                out_channels=b3_channels[0],
                kernel_size=1,
                stride=1,
            ),
            BasicConv2d(
                in_channels=b3_channels[0],
                out_channels=b3_channels[1],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
        self.branch_4 = nn.Sequential(
            MaxPool2d(3, 1),
            BasicConv2d(
                in_channels=in_channel,
                out_channels=b4_channel,
                kernel_size=1,
                stride=1,
                padding=1,
            ),
        )

    def _forward(self, x):
        out_b1 = self.branch_1(x)
        out_b2 = self.branch_2(x)
        out_b3 = self.branch_3(x)
        out_b4 = self.branch_4(x)

        return [out_b1, out_b2, out_b3, out_b4]

    def forward(self, x):
        outs = self._forward(x)
        return torch.concat(outs, 1)


class InceptionV1(nn.Module):
    def __init__(self, classes) -> None:
        super(InceptionV1, self).__init__()
        self.max_pool = MaxPool2d(3, stride=2)
        self.first_convolution = nn.Sequential(
            BasicConv2d(3, 64, 7, stride=2),
            self.max_pool,
            BasicConv2d(64, 192, 3, stride=1),
            self.max_pool,
        )
        self.inception_block_3a = Inception_block(192, 64, (96, 128), (16, 32), 32)
        self.inception_block_3b = Inception_block(256, 128, (128, 192), (32, 96), 64)

        self.inception_block_4a = Inception_block(480, 192, (96, 208), (16, 48), 64)
        self.inception_block_4b = Inception_block(512, 160, (112, 224), (24, 64), 64)
        self.inception_block_4c = Inception_block(512, 128, (128, 256), (24, 64), 64)
        self.inception_block_4d = Inception_block(512, 112, (144, 288), (32, 64), 64)
        self.inception_block_4e = Inception_block(528, 256, (160, 320), (32, 128), 128)

        self.inception_block_5a = Inception_block(832, 256, (160, 320), (32, 128), 128)
        self.inception_block_5b = Inception_block(832, 384, (192, 384), (48, 128), 128)

        self.avg_pool = AvgPool2d(kernel_size=5, stride=1)
        self.dropout = Dropout2d(p=0.4)
        self.fc = nn.Linear(1024, classes, bias=False)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.first_convolution(x)
        x = self.inception_block_3a(x)
        x = self.inception_block_3b(x)
        x = self.max_pool(x)
        x = self.inception_block_4a(x)
        x = self.inception_block_4b(x)
        x = self.inception_block_4c(x)
        x = self.inception_block_4d(x)
        x = self.inception_block_4e(x)
        x = self.max_pool(x)
        x = self.inception_block_5a(x)
        x = self.inception_block_5b(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = InceptionV1()
