"""
Contains functions for displaying model performance during training and evalution for classification tasks
"""
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_mat(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader,
                       class_names: list, test_targets: list):
    """
    Plot the confusion matrix for a given batch of test data using the given model
    :param model: a pytorch model for classification of class nn.Module to be evaluated
    :param test_dl: the pytorch dataloader of the test data
    :param class_names: the names of class names for the data
    :param test_targets: the targets of the test data
    """
    model.eval()
    predictions = []
    with torch.inference_mode():
        for X, y in test_dl:
            logit = model(X)
            pred = torch.softmax(logit, dim=1).argmax(dim=1)  # get the class predictions and add them to the list
            predictions.append(pred)

    pred_tensor = torch.cat(predictions)

    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=pred_tensor,
                             target=torch.tensor(test_targets))

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=class_names,  # turn the row and column labels into class names
        figsize=(10, 7)
    )
    plt.show()


def plot_loss_curve(res: dict[str, list[float]], epochs: range):
    """
    Display the loss and accuracy curves over given range of epochs for train and test data
    :param res: a dict in the form:
        {"train_loss": [train loss values for each epoch],
        "train_acc": [train accuracy values for each epoch],
        "test_loss": [test loss values for each epoch],
        "test_acc": [test accuracy values for each epoch]}
    :param epochs: the range of epochs to display
    """
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, res["train_loss"], label="train_loss")
    plt.plot(epochs, res["test_loss"], label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, res["train_acc"], label="train_accuracy")
    plt.plot(epochs, res["test_acc"], label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()
