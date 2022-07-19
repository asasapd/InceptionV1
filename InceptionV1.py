from typing import Tuple
from torch import nn, relu
import torch
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Dropout2d

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

class AuxClassifier(nn.Module):
    # • Average_pooling 5x5 s=3
    # • A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation.
    # • A fully connected layer with 1024 units and rectified linear activation.
    # • A dropout layer with 70% ratio of dropped outputs.
    # • A linear layer with softmax loss as the classifier 
    def __init__(self, in_channel: int, classes: int, fc_in: int) -> None:
        super(AuxClassifier, self).__init__()
        self.avgPool = AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
        self.fc = nn.Linear(fc_in, 1024)
        self.activation = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.7),
            nn.Linear(1024, classes)
        )
        
        
    def forward(self, x):
        N = x.shape[0]
        x = self.avgPool(x)
        x = self.conv(x)
        
        x = x.reshape(N, -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
    
class InceptionV1(nn.Module):
    def __init__(self, classes) -> None:
        super(InceptionV1, self).__init__()
        self.max_pool = MaxPool2d(3, stride=2, padding=1)
        self.first_convolution = nn.Sequential(
            BasicConv2d(3, 64, 7, stride=2, padding=3),
            self.max_pool,
            BasicConv2d(64, 64, 1, stride=1, padding=1),
            BasicConv2d(64, 192, 3, stride=1),
            self.max_pool,
        )
        self.inception_block_3a = Inception_block(192, 64, (96, 128), (16, 32), 32)
        self.inception_block_3b = Inception_block(256, 128, (128, 192), (32, 96), 64)

        self.inception_block_4a = Inception_block(480, 192, (96, 208), (16, 48), 64)
        #  -> 4x4x512
        self.aux_1 = AuxClassifier(in_channel=512, classes=classes, fc_in=2048)
        self.inception_block_4b = Inception_block(512, 160, (112, 224), (24, 64), 64)
        self.inception_block_4c = Inception_block(512, 128, (128, 256), (24, 64), 64)
        self.inception_block_4d = Inception_block(512, 112, (144, 288), (32, 64), 64)
        # Average_pooling 5x5 s=3 -> 4x4x528
        self.aux_2 = AuxClassifier(in_channel=528, classes=classes, fc_in=2048)
        self.inception_block_4e = Inception_block(528, 256, (160, 320), (32, 128), 128)

        self.inception_block_5a = Inception_block(832, 256, (160, 320), (32, 128), 128)
        self.inception_block_5b = Inception_block(832, 384, (192, 384), (48, 128), 128)
      
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024 * 7 * 7, classes)
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.first_convolution(x)
        x = self.inception_block_3a(x)
        x = self.inception_block_3b(x)
        x = self.max_pool(x)
        x = self.inception_block_4a(x)
        aux_1 = self.aux_1(x)
        x = self.inception_block_4b(x)
        x = self.inception_block_4c(x)
        x = self.inception_block_4d(x)
        aux_2 = self.aux_2(x)
        x = self.inception_block_4e(x)
        x = self.max_pool(x)
        x = self.inception_block_5a(x)
        x = self.inception_block_5b(x)
        
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        if self.training == True:
            return [x, aux_1, aux_2]
        return x