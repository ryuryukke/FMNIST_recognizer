import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        # input_dim = args.input_dim
        input_dim = 28 * 28
        fc1_dim = args.fc1
        fc2_dim = args.fc2

        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, fc1_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc2_dim, 10),
            nn.ReLU()
        )
    

    def forward(self, x):
        # logits = Value before being put into the softmax function
        logits = self.linear_relu_stack(x)
        return logits


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)
    

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        logits = self.fc2(x)
        return logits




