"""
Contains functions for training or evaluating a model for a single epoch
"""
import torch


def train_step(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        l_funct: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device
):
    """
    Train the given pytorch model for a single epoch; goes through forward pass, loss calculation, optimizer step
    :param model: pytorch model to train
    :param train_dl: pytorch dataloader containing the training data
    :param l_funct: pytorch loss function
    :param optim: pytorch optimizer function
    :param device: target device, either cuda or cpu
    :return: a tuple of the form (train loss, train accuracy)
    """
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dl):
        X, y = X.to(device), y.to(device)  # send data to cpu or cuda depending on given device
        y_pred = model(X)
        loss = l_funct(y_pred, y)
        train_loss += loss  # accumulate loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # the predicted class is the one with the max value after softmax
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # accumulate loss and accuracy for every batch, then get avg loss and accuracy per batch
    train_loss /= len(train_dl)
    train_acc /= len(train_dl)

    return train_loss, train_acc


def test_step(
        model: torch.nn.Module,
        test_dl: torch.utils.data.DataLoader,
        l_funct: torch.nn.Module,
        device: torch.device
):
    """
    Evaluate the given pytorch model for a single epoch; goes through forward pass, loss calculation
    :param model: pytorch model to train
    :param test_dl: pytorch dataloader containing the test data
    :param l_funct: pytorch loss function
    :param device: target device, either cuda or cpu
    :return: a tuple of the form (test loss, test accuracy)
    """
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)  # send data to cpu or cuda depending on given device
            test_pred = model(X)
            test_loss += l_funct(test_pred, y)  # accumulate loss

            # the predicted class is the one with the max value after softmax
            test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += (test_pred_class == y).sum().item() / len(test_pred)

        # accumulate loss and accuracy for every batch, then get avg loss and accuracy per batch
        test_loss /= len(test_dl)
        test_acc /= len(test_dl)

    return test_loss, test_acc


def train(
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        test_dl: torch.utils.data.DataLoader,
        l_funct: torch.nn.Module,
        optim: torch.optim.Optimizer,
        epochs: int,
        device: torch.device
):
    """
    Train the given model and store the metrics for each epoch for later use
    :param model: pytorch model to train
    :param train_dl: pytorch dataloader containing the train data
    :param test_dl: pytorch dataloader containing the test data
    :param l_funct: pytorch loss function
    :param optim: pytorch optimizer function
    :param epochs: the number of times to go through the training loop and update parameters
    :param device: target device, either cuda or cpu
    :return: a dict in the form:
        {"train_loss": [train loss values for each epoch],
        "train_acc": [train accuracy values for each epoch],
        "test_loss": [test loss values for each epoch],
        "test_acc": [test accuracy values for each epoch]}
    """
    res = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for e in range(epochs):
        # get test loss and accuracy, add to res
        train_loss, train_acc = train_step(model, train_dl, l_funct, optim, device)
        res["train_loss"].append(float(train_loss.detach()))  # convert scalar to float
        res["train_acc"].append(train_acc)

        # get test loss and accuracy, add to res
        test_loss, test_acc = test_step(model, test_dl, l_funct, device)
        res["test_loss"].append(float(test_loss.detach()))  # convert scalar to float
        res["test_acc"].append(test_acc)

        # print results of each training epoch
        print(f"\nINFO:    Epoch: {e} Train loss: {train_loss:.5f}, Train acc: {train_acc:.2f}% | "
              f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

    return res
