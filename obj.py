import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout(p = 0.2)
        # nn.Conv2d applies a 2D convolution 
        # syntax torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1 = nn.Conv2d(3, 96, 5) 
        self.conv1A = nn.Conv2d(96, 96, 1)
        # nn.MaxPool2d applies a 2D maximum pooling
        # syntax  torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,return_indices=False,ceil_mode=False)
        self.pool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.conv2 = nn.Conv2d(96, 192, 5)
        self.conv2A = nn.Conv2d(192, 192, 1)
        self.conv3 = nn.Conv2d(192, 192, 3)
        self.conv4 = nn.Conv2d(192, 192, 1)
        self.conv5 = nn.Conv2d(192, 10, 1)
        self.AveragePool = nn.AdaptiveAvgPool2d((1,1))
        # nn.Linear applies a normal linear transformation of y = xA + b

    # define forward propagation algorithm True
    def forward(self, x):
        # if inplace is true, it will change the content of the given Tensor directly instead of make a copy of it
        x = self.input_dropout(x)
        x = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv1A(x), inplace=False)
        x = self.dropout(self.pool(x))
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv2A(x), inplace=False)
        x = self.dropout(self.pool(x))
        x = F.relu(self.conv3(x), inplace=False)
        # x = self.AveragePool(F.relu(self.conv4(x), inplace=False))
        x = F.relu(self.conv4(x), inplace=False)
        x = self.AveragePool(F.relu(self.conv5(x), inplace=False))
        # torch.flatten(input, start_dim, end_dim=-1)
        # start_dim default is 0
        x = torch.flatten(x, 1)

        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout(p = 0.2)
        # nn.Conv2d applies a 2D convolution 
        # syntax torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1 = nn.Conv2d(3, 64, 3) 
        # nn.MaxPool2d applies a 2D maximum pooling
        # syntax  torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,return_indices=False,ceil_mode=False)
        self.pool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

    # define forward propagation algorithm
    def forward(self, x):
        # if inplace is true, it will change the content of the given Tensor directly instead of make a copy of it
        x = self.input_dropout(x)
        x = F.relu(self.conv1(x), inplace=False)
        x = self.dropout(self.pool(x))
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        x = self.dropout(self.pool(x))
        # torch.flatten(input, start_dim, end_dim=-1)
        # start_dim default is 0
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

if __name__ == '__main__':
    n = Net()
    x = torch.randn((8, 3, 32, 32))

    print(n(x).shape)