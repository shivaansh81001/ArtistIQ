import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_basic(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1,32,stride=1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,stride=1,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(64*7*7,128)
        self.lin2 = nn.Linear(128,num_classes)
        self.dropout = nn.Dropout(0.5)
        self.probs = None

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2,2)
        x = x.view(x.size(0),-1)
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.lin2(x)
        self.probs = F.softmax(x,dim=1)
        return x

    def return_probs(self):
        return self.probs
