import torch


class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 10)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(64, 128, 7)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(128 * 24 * 24, 1024)
        self.linear2 = torch.nn.Linear(1024, 128)
    
    def forward(self, x):
        x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool2(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 128 * 24 * 24)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x