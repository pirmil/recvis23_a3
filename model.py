import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 250

def freeze_parameters(model: nn.Module, layers_to_finetune: str) -> None:
    if layers_to_finetune=="classifier":
        # Freeze features parameters
        for param in model.features.parameters():
            param.requires_grad = False
    elif layers_to_finetune=="last_layer":
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    elif layers_to_finetune=="whole_model":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError(f"This fine-tuning method is not implemented.")

class IncrementalTrainedModel(nn.Module):
    def __init__(self, model_path: str, class_name: str, layers_to_finetune: str) -> None:
        """
        Args:
            model_path (str): Path to the model to continue training
            class_name (str): Class of the model to continue training
            layers_to_finetune (str): The layers of the model that must be trained
        """
        super(IncrementalTrainedModel, self).__init__()
        if class_name == 'finetuned_AlexNet':
            self.base_model = FineTunedAlexNetClassifier(layers_to_finetune)
        elif class_name == 'finetuned_VGG':
            self.base_model = FineTunedVGGClassifier(layers_to_finetune)
        elif class_name == 'finetuned_ResNet':
            self.base_model = FineTunedResNetClassifier(layers_to_finetune)
        elif class_name == 'small_VGG':
            self.base_model = SmallVGGClassifier()
        elif class_name == 'basic_cnn':
            self.base_model = Net()
        else:
            raise ValueError(f"Unsupported model class: {class_name}")
        checkpoint = torch.load(model_path)
        self.base_model.load_state_dict(checkpoint)
        freeze_parameters(self.base_model, layers_to_finetune)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class FineTunedAlexNetClassifier(nn.Module):
    def __init__(self, layers_to_finetune: str) -> None:
        super(FineTunedAlexNetClassifier, self).__init__()
        # Load pre-trained AlexNet
        alexnet = models.alexnet(weights='DEFAULT')
        
        freeze_parameters(alexnet, layers_to_finetune)
        
        # Extract features and average pooling layers
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = alexnet.classifier
        n_input = alexnet.classifier[-1].in_features
        # Modify the last fully connected layer for our task
        self.classifier[-1] = nn.Linear(in_features=n_input, out_features=nclasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FineTunedResNetClassifier(nn.Module):
    def __init__(self, layers_to_finetune: str) -> None:
        super(FineTunedResNetClassifier, self).__init__()
        # Load pre-trained ResNet
        resnet = models.resnet50(weights='DEFAULT')

        freeze_parameters(resnet, layers_to_finetune)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        n_input = resnet.fc.in_features
        self.fc = nn.Linear(in_features=n_input, out_features=nclasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FineTunedVGGClassifier(nn.Module):
    def __init__(self, layers_to_finetune: str) -> None:
        super(FineTunedVGGClassifier, self).__init__()
        # Load pre-trained VGG-16
        vgg16 = models.vgg16(weights='DEFAULT')
        
        freeze_parameters(vgg16, layers_to_finetune)
        
        # Extract features and average pooling layers
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        n_input = vgg16.classifier[-1].in_features
        # Modify the last fully connected layer for our task
        self.classifier[-1] = nn.Linear(in_features=n_input, out_features=nclasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGGFeatures(nn.Module):
    def __init__(self) -> None:
        super(VGGFeatures, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


class SmallVGGClassifier(nn.Module):
    def __init__(self, num_classes: int = nclasses, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = VGGFeatures()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class MultiModel(nn.Module):
    def __init__(self, num_classes=nclasses):
        super(MultiModel,self).__init__()
        
        self.resnet = models.resnet50(weights='DEFAULT')
        self.vgg = models.vgg16_bn(weights='DEFAULT')

        n_input = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(in_features=n_input, out_features=256)

        n_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=n_input, out_features=256)

        for param in self.resnet.parameters():
            param.requires_grad = True
        for param in self.vgg.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(256 * 2, nclasses)


      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xres = self.resnet(x)
        xvgg = self.vgg(x)
        x = torch.cat((xres,xvgg), 1)
        return self.fc(x)
