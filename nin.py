import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod 
    def forward(self, input):
        # keep the full precision activation for back-propagation
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    @staticmethod 
    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinAConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinAConv2d, self).__init__()
        self.layer_type = 'BinAConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        # full precision BatchNorm2d
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)

        # intialize the weights to all ones
        # what is the weight.data here??
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # first do batch normalization
        x = self.bn(x)
        # binarize the input activation
        x, mean = BinActive.apply(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        # full precision BatchNorm2d
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        # intialize the weights to all ones
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class XNORNet(nn.Module):
    def __init__(self, num_classes=10):
        super(XNORNet, self).__init__()
        self.num_classes = num_classes
        self.xnor = nn.Sequential(
                # in_channel, out_channel
                # the first batch norm layer
                # eps is a value added to the denominator for numerical stability
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),

                BinAConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinAConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinAConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                BinAConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinAConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinAConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinAConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                # the second batch norm layer
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    # clamp data into the range [0.01,)
                    m.weight.data.clamp_(min=0.01)
        x = self.xnor(x)
        #.view is similar to reshape function
        x = x.view(x.size(0), self.num_classes)
        return x


class BWNNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BWNNet, self).__init__()
        self.num_classes = num_classes
        self.bwn = nn.Sequential(
            # in_channel, out_channel
            # the first batch norm layer
            # eps is a value added to the denominator for numerical stability
            # input 32*32
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 16*16
            BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            # 8*8
            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(192,  self.num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            # pooling operation preserve the number of input channels
            # 1*1
         )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    # clamp data into the range [0.01,)
                    m.weight.data.clamp_(min=0.01)

        x = self.bwn(x)
        #.view is similar to reshape function
        x = x.view(x.size(0), self.num_classes)
        return x



class normal(nn.Module):
    def __init__(self, num_classes=10):
        super(normal, self).__init__()
        self.num_classes = num_classes
        self.bwn = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(160, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 16*16
            nn.Conv2d( 96, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            # 8*8
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,  self.num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            )

    def forward(self, x):
        x = self.bwn(x)
        x = x.view(x.size(0), self.num_classes)
        return x