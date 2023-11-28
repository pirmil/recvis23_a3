import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 224 x 224 in size because this is the expected image size
# for the fine-tuned models AlexNet, ResNet and VGG
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet

# Validation and test data
data_transforms_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Training data: data augmentation with RandomResizedCrop and RandomHorizontalFlip
data_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


## FOR INCEPTION ONLY (inception expects 299x299 images)
# Validation and test data
data_transforms_valid_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.CenterCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Training data: data augmentation with RandomResizedCrop and RandomHorizontalFlip
data_transforms_train_inception = transforms.Compose([
    transforms.RandomResizedCrop((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
