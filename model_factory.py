from __future__ import annotations
import torch.nn as nn
from torchvision import transforms
from model import SmallVGGClassifier, FineTunedVGGClassifier, FineTunedAlexNetClassifier, IncrementalTrainedModel, FineTunedResNetClassifier, Net, MultiModel, FineTunedDenseNetClassifier
from data import data_transforms_train, data_transforms_valid

acceptable_models = {"small_VGG", "finetuned_VGG", "finetuned_AlexNet", "finetuned_ResNet", "incremental_model", "basic_cnn", "multi_model", "finetuned_DenseNet"}

class ModelFactory:
    """
    Class to instantiate the model and the transforms that go with it.
    """
    def __init__(self, model_name: str, layers_to_finetune: str, model_path: str, class_name: str) -> None:
        self.model_name = model_name
        self.model = self.init_model(layers_to_finetune, model_path, class_name)
        self.transforms_train = self.init_transforms_train()
        self.transforms_valid = self.init_transforms_valid()

    def init_model(self, layers_to_finetune: str, model_path: str, class_name: str) -> nn.Module:
        if self.model_name == "small_VGG":
            return SmallVGGClassifier()
        elif self.model_name == "finetuned_VGG":
            return FineTunedVGGClassifier(layers_to_finetune)
        elif self.model_name == 'finetuned_AlexNet':
            return FineTunedAlexNetClassifier(layers_to_finetune)
        elif self.model_name == 'finetuned_ResNet':
            return FineTunedResNetClassifier(layers_to_finetune)
        elif self.model_name == 'incremental_model':
            return IncrementalTrainedModel(model_path, class_name, layers_to_finetune)
        elif self.model_name == 'basic_cnn':
            return Net()
        elif self.model_name == 'multi_model':
            return MultiModel()
        elif self.model_name == 'finetuned_DenseNet':
            return FineTunedDenseNetClassifier(layers_to_finetune)
        else:
            raise NotImplementedError("Model not implemented")
        
    def init_transforms_train(self) -> transforms.Compose:
        if self.model_name in acceptable_models:
            return data_transforms_train
        else:
            raise NotImplementedError("Transform not implemented")       

    def init_transforms_valid(self) -> transforms.Compose:
        if self.model_name in acceptable_models:
            return data_transforms_valid
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self) -> nn.Module:
        return self.model

    def get_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        return self.transforms_train, self.transforms_valid

    def get_all(self) -> tuple[nn.Module, transforms.Compose, transforms.Compose]:
        return self.model, self.transforms_train, self.transforms_valid
