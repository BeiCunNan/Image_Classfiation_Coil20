import torch
from torch import nn
import torch.nn.functional as F


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 20)
        )

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch_size
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(20, 40, kernel_size=2, stride=(1, 1))
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(640, 160)
        self.fc2 = torch.nn.Linear(160, 20)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))  # conv1+pooling
        x = F.relu(self.pooling(self.conv2(x)))  # conv2+pooling
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc1(x)  # FNN
        x = self.fc2(x)
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
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)  # 88 = 24x3 + 16

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
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
        )
        self.fnn = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 20)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(batch_size, -1)
        x = self.fnn(x)
        return x
