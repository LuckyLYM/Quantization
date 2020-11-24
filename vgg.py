import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            # check out what is batchnorn
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        # binActive
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x



class BWNNet(nn.Module):

    def __init__(self, num_classes=10):
        super(BWNNet, self).__init__()
        self.num_classes = num_classes


        self.features = nn.Sequential(
            # full preicsion conv1 32*32
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # binary conv2
            BinConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 16*16

            # binary conv3
            BinConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # binary conv4
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # 8*8

            # binary conv5
            BinConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # binary conv6
            BinConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # 4*4
        )
        self.classifier = nn.Sequential(
            BinConv2d(512 * 4 * 4, 1024, Linear=True),
            BinConv2d(1024, 1024, Linear=True),
            nn.BatchNorm1d(1024, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 4 * 4)
        x = self.classifier(x)
        return x


class normal(nn.Module):

    def __init__(self, num_classes=10):
        super(normal, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8*8

            # binary conv5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 4*4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*4*4, 1024),
            nn.ReLU(inplace=True),
    
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 4 * 4)
        x = self.classifier(x)
        return x