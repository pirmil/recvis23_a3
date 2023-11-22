import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 250

class FineTunedVGGClassifier(nn.Module):
    def __init__(self):
        super(FineTunedVGGClassifier, self).__init__()
        
        # Load pre-trained VGG-16
        vgg16 = models.vgg16(weights='DEFAULT')
        
        # Extract features and average pooling layers
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        
        # Freeze all layers except the last fully connected layer
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Modify the last fully connected layer for your task
        self.flatten = nn.Flatten()
        self.classifier = ClassificationHead()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, n_input=512*7*7):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(n_input, nclasses)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for layer in [self.fc1]:
            layer.bias.data.zero_()
            layer.weight.data.uniform_(-initrange, initrange)   

    def forward(self, x):
        return self.fc1(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
