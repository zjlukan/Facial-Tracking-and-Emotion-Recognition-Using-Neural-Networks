"""
Contains functionality for creating PyTorch dataloaders used in image classification tasks
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS
):
    """
    Create training and testing dataloaders from train and test data directories
    :param train_dir: train data directory path
    :param test_dir: test data directory path
    :param train_transform: the transform that is applied to the train data (torchvision transforms)
    :param test_transform: the transform that is applied to the test data (torchvision transforms)
    :param batch_size: number of samples per batch
    :param num_workers: number of subprocesses used to load data in parallel
    :return: a tuple of the form (train dataloader, test dataloader, class names, test data targets)
    """
    # load the data into datasets from the given directories, extract list of class names
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    class_names = train_data.classes
    test_targets = test_data.targets

    # convert datasets into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names, test_targets


def get_classes(
        directory: str
):
    """
    Return list of classes for the given data
    :param directory: data directory path
    :return: list of class names
    """
    dataset = datasets.ImageFolder(directory)
    return dataset.classes


def get_aug_dataset(
        data_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    """
    create an augmented dataset
    :param data_dir:
    :param transform:
    :param batch_size:
    :param num_workers:
    :return:
    """
    data = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


