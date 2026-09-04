"""
Contains different utility functions for pytorch model creation
"""
import torch
from torch import nn
from pathlib import Path


def init_weights(model: torch.nn.Module):
    """
    Given a CNN, initialize the weights using He initialization
    :param model: a CNN model with nn.Conv2d layers
    """
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')

        # set bias to 0
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)


def save_model(
        model: torch.nn.Module,
        model_name: str,
        directory: str
):
    """
    Given a model, save it to the given directory
    :param directory: the path to the directory inside which the model will be saved
    :param model: the pytorch model to be saved to file
    :param model_name: the name that the model will be saved under
    """
    model_path = Path(directory)

    # create directory if it does not exist
    model_path.mkdir(parents=True, exist_ok=True)
    save_path = model_path / model_name
    torch.save(obj=model.state_dict(), f=save_path)


def load_model(
        model: torch.nn.Module,
        path: str
):
    """
    Load a model from the given state dict
    :param path: the path to the state dict file
    :param model: a pytorch model with the same architecture as the saved model
    :return: the model with the loaded state
    """
    model_path = Path(path)
    model.load_state_dict(torch.load(f=model_path))

    return model

