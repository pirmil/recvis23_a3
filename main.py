from __future__ import annotations
from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from matplotlib import pyplot as plt
from model import IncrementalTrainedModel
from model_factory import ModelFactory
from torch.optim.lr_scheduler import ExponentialLR, PolynomialLR, ReduceLROnPlateau, LambdaLR


def opts() -> ArgumentParser:
    """Option Handling Function."""
    parser = ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="finetuned_VGG",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiment/model_best.pth",
        metavar="mp",
        help="The path to the model whose training is being resumed (only works with an incremental model)",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="finetuned_VGG",
        metavar="CLN",
        help="The model name of the model whose training is being resumed (only works with an incremental model)",
    )
    parser.add_argument(
        "--layers_to_finetune",
        type=str,
        default="whole_model",
        metavar="tbf",
        help="Finetune the last layer or the full classifier or the full model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="B",
        help="Input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Maximum learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=5e-5,
        metavar="WLR",
        help="Start learning rate if there is a warm up (default: 5e-5)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.2,
        metavar="W",
        help="Ratio of the epochs that must be used as warm up (default: 20%)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        metavar="lrS",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        metavar="wd",
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="How many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="Folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        metavar="NW",
        help="Number of workers for data loading (default: 2)",
    )
    parser.add_argument(
        "--convert_val_to_train",
        type=float,
        default=0,
        metavar="IT",
        help="Removes the requested percentage of the validation set and adds it to the training set",
    )
    args = parser.parse_args()
    return args


def create_loaders(args: ArgumentParser, data_transforms_train: transforms.Compose, data_transforms_valid: transforms.Compose) -> tuple[DataLoader, DataLoader]:
    """
    Loads the validation and training data. It allows to convert validation data into training data.
    """
    train_dataset = datasets.ImageFolder(args.data + "/train_images", transform=data_transforms_train)
    val_dataset = datasets.ImageFolder(args.data + "/val_images", transform=data_transforms_valid)
    num_val_samples = len(val_dataset)
    num_train_val_samples = int(args.convert_val_to_train * num_val_samples)

    val_indices_train = torch.randperm(num_val_samples)[:num_train_val_samples]
    subset_val_dataset_train = Subset(val_dataset, val_indices_train)

    val_indices_val = list(set(range(num_val_samples)) - set(val_indices_train.tolist()))
    subset_val_dataset_val = Subset(val_dataset, val_indices_val)

    combined_dataset = ConcatDataset([train_dataset, subset_val_dataset_train])

    train_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        subset_val_dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"The training dataset contains {len(train_loader.dataset)} images")
    print(f"The validation dataset contains {len(val_loader.dataset)} images")
    return train_loader, val_loader


def plot_losses(train_losses: list[float], val_losses: list[float], args: ArgumentParser) -> None:
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(args.experiment, 'loss_plot.png'))
    plt.show()

def plot_accuracies(train_accuracies: list[float], val_accuracies: list[float], args: ArgumentParser) -> None:
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.savefig(os.path.join(args.experiment, 'accuracy_plot.png'))
    plt.show()

def plot_learning_rates(learning_rates: list[float], args: ArgumentParser) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('Evolution of the learning rate over the epochs')
    plt.savefig(os.path.join(args.experiment, 'learning_rate.png'))
    plt.show()   

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    use_cuda: bool,
    epoch: int,
    args: ArgumentParser,
) -> tuple[float, float]:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (ArgumentParser): Arguments parsed from command line
    
    Returns:
        Training loss (float): training loss
        Training accuracy (float): training accuracy
    """
    model.train()
    training_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss(reduction="mean")
        if args.model_name=='finetuned_inception':
            # Only for inception, auxiliary output
            output, aux_output = model(data)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            output = model(data)
            loss = criterion(output, target)
        training_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    training_loss /= len(train_loader.dataset)
    return training_loss, 100.0 * correct / len(train_loader.dataset)


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> tuple[float, float]:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        Validation loss (float): validation loss
        Validation accuracy (float): validation accuracy
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss, 100.0 * correct / len(val_loader.dataset)

def parameters_to_optimize(model: nn.Module) -> list:
    params_to_update = []
    print("Parameters to update:")
    list_to_optimize = model.base_model.named_parameters() if isinstance(model, IncrementalTrainedModel) else model.named_parameters()
    for name, param in list_to_optimize:
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t{name}")
    return params_to_update

def warmup_lambda(epoch: int, warmup_duration: int, args: ArgumentParser) -> float:
    if warmup_duration==1 and epoch==0: 
        return args.warmup_lr / args.lr
    elif epoch < warmup_duration:
        return ((args.lr - args.warmup_lr) * epoch / (warmup_duration - 1)  + args.warmup_lr) / args.lr
    else:
        return 1

def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms_train, data_transforms_valid = ModelFactory(args.model_name, args.layers_to_finetune, args.model_path, args.class_name).get_all()
    if use_cuda:
        print("Using GPU")
        model = model.to(device)
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader, val_loader = create_loaders(args, data_transforms_train, data_transforms_valid)

    # Setup optimizer
    optimizer = optim.SGD(parameters_to_optimize(model), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Setup learning rate scheduler
    if args.lr_scheduler=='constant':
        lr_lambda = lambda epoch: 1
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.lr_scheduler=='reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=1, factor=0.2)
    elif args.lr_scheduler=='polynomial_decay':
        scheduler = PolynomialLR(optimizer=optimizer, total_iters=args.epochs-warmup_duration, power=2)
    elif args.lr_scheduler=='exponential_decay':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=0.95)
    elif args.lr_scheduler=='warmup_constant':
        warmup_duration = int(args.warmup_ratio * args.epochs)
        print(f"Warmup epochs: {warmup_duration} / {args.epochs}")
        lr_lambda = lambda epoch: warmup_lambda(epoch, warmup_duration, args)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)  
    else:
        raise NotImplementedError(f"The specified learning rate scheduler is not implemented")
    

    # Lists to store training and validation losses, and accuracies
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        train_loss, train_acc = train(model, optimizer, train_loader, use_cuda, epoch, args)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        # validation loop
        val_loss, val_acc = validation(model, val_loader, use_cuda)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
        # Step of the learning rate
        learning_rates.append(optimizer.param_groups[0]['lr'])
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses, args)

    # Plot the training and validation accuracies
    plot_accuracies(train_accuracies, val_accuracies, args)

    plot_learning_rates(learning_rates, args)

if __name__ == "__main__":
    main()
