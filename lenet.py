from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinConv2d(nn.Module): 
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            if self.previous_conv:
                self.bn = nn.BatchNorm2d(int(input_channels/size), eps=1e-4, momentum=0.1, affine=True)
            else:
                self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x

class BWNNet(nn.Module):
    def __init__(self,num_classes=10):
        self.num_classes = num_classes
        super(BWNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)
        # 28*28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 14*14
        self.bin_conv2 = BinConv2d(20, 50, kernel_size=5, stride=1, padding=0)
        # 10*10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 50*5*5
        # but why 4 by 4 here???
        self.bin_ip1 = BinConv2d(50*5*5, 500, Linear=True,
                previous_conv=True, size=5*5)
        # full connected
        self.ip2 = nn.Linear(500, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.bin_conv2(x)
        x = self.pool2(x)
        x = self.bin_ip1(x)
        x = self.ip2(x)
        return x


class normal(nn.Module):
    def __init__(self,num_classes=10):
        super(normal, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.BatchNorm2d(20, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(20, 50, kernel_size=5),
            nn.BatchNorm2d(50, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(50*5*5, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50*5*5)
        x = self.classifier(x)
        return x