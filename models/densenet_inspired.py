import torch
import torch.nn as nn



class DensenetInspired(nn.Module):
    def __init__(self):
        super(DensenetInspired, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        # This part is Densenet inspired...recycle use of BN, RELU, CONV repeatedly
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3= nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        self.flat = nn.Flatten()
        self.lin1 = nn.LazyLinear(out_features=1024)
        self.fc1 = nn.Linear(1024, 80)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.mp1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        out = self.flat(out)
        out = self.lin1(out)
        out = self.fc1(out)

        return out