import torch
from torch import nn
import torch.nn.functional as F


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 10, 10, 2),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(10, 20, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(20, 20, 4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(20, 20, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(5 * 5 * 20, 100),
            nn.Sigmoid(),
            nn.Linear(100, 60),
            nn.Sigmoid(),
            nn.Linear(60, 20)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.convs(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=20, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=20)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.convs(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c对应的是dim=1


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)  # 与conv1 中的10对应
        self.incep2 = InceptionA(in_channels=20)  # 与conv2 中的20对应

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 20)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=17)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.convs = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )
        self.fnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256),
            nn.Linear(256, 20),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.convs(x),
        x = x[0]
        x = x.view(batch_size, -1)
        x = self.fnn(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, padding=None, first=False):
        super(Bottleneck, self).__init__()
        if stride is None:
            stride = [1, 1, 1]
        if padding is None:
            padding = [0, 1, 0]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2]),
            nn.BatchNorm2d(out_channels * 4)
        )

        if (first):
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1]),
                nn.BatchNorm2d(out_channels * 4)
            )
        else:
            self.res = nn.Sequential()

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.res(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=17),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2)
        ) # --> 3,4,3,6
        self.conv2 = self._make_layer(Bottleneck, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3, 64)
        self.conv3 = self._make_layer(Bottleneck, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4, 128)
        self.conv4 = self._make_layer(Bottleneck, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6, 256)
        self.conv5 = self._make_layer(Bottleneck, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3, 512)

        self.convs = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

        self.fnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256),
            nn.Linear(256, 20),
        )

    def _make_layer(self, Bottleneck, strides, paddings, out_channels):
        layers = []
        flag = True
        for i in range(len(strides)):
            layers.append(Bottleneck(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.convs(x)
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.fnn(x)
        return x


class EfficientNet(torch.nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()

    def forward(self, x):
        return x
