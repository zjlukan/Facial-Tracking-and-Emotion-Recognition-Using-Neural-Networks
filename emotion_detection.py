import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import cv2
import os
import random
from PIL import Image
from pathlib import Path
from torchvision import datasets, transforms
import torchinfo
from torchinfo import summary
import multiprocessing as mp
import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


class TinyVGG_model(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int, dropout_rate=0.0):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=out_shape)
            # in_features shape according to shape changes to input data from conv and pooling layers
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


def init_weights(m: torch.nn.Module):
    """
    Given a CNN, initialize the weights using He initialization
    :param m: a CNN model with nn.Conv2d layers
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def read_face(image, box, detected, model):
    """
    Given the bounding box of the face, write the bounding box and the emotion to the screen. If not face detected,
    write "No face detected!"
    :param model: a loaded model that is of class nn.Module
    :param detected: indicates if a face was detected in this frame
    :param image: a 3-channel image
    :param box: bounding box in the format [x, y, width, height] (or [] if the face is not detected)
    :return: nothing
    """
    if len(box) == 1:
        box = box[0]

    # display text if no face detected
    if not detected:
        cv2.putText(
            image,
            "No face detected!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

    # display a bounding box and feed the cropped image to the model
    else:
        cv2.rectangle(
            image,
            (box[0], box[1]),
            (box[0] + box[2], box[1] + box[3]),
            (255, 255, 0),
            3
        )

        # process the image into a 64x64 grayscale tensor to be fed into the model
        cropped_img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        data_transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        transformed_img = data_transform(cropped_img)

        # pass the processed image to the model to get a prediction
        model.eval()
        with torch.inference_mode:
            pred = model(transformed_img)
            class_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        cv2.putText(image, str(class_pred), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA)

    # display the final image
    cv2.imshow("webcam", image)
    print(box)


def track_face():
    """
    Capture footage from webcam and calculate a bounding box which will be used to crop the image before feeding it
    into the CNN. Uses a cascade classifier to initially locate the face and a KCF tracker to update the bounding box.
    Relocates the face every 50 frames in order to prevent drift
    """
    model_path = Path("Models/emotion_classifier_0.pth")
    loaded_model = TinyVGG_model(in_shape=1, out_shape=7, hidden_units=10)
    loaded_model.load_state_dict(torch.load(f=model_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    # find the face in the first frame and get a bounding box
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    still_frames, f = cap.read()
    if not still_frames:
        print("Can't read frames from capture")
        exit()
    frame_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    # do not initialize the tracker until the cascade classifier has detected a face
    detected = False
    tracker = cv2.TrackerKCF_create()
    if len(face) != 0:
        tracker.init(f, face[0])
        detected = True

    x = 1
    while True:
        still_frames, f = cap.read()
        if not still_frames:
            print("no more frames")
            break
        s = False
        if detected:
            s, box = tracker.update(f)

        # if there is a face detected and the tracker updates properly, process the frame
        if s:
            print(len(box))
            read_face(f, box, detected)

        # if not, relocate the face
        # relocate the face every 50 frames to prevent drift
        if not s or x % 50 == 0:
            tracker = cv2.TrackerKCF_create()
            frame_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            detected = False
            if len(face) != 0:
                tracker.init(f, face[0])
                detected = True
            read_face(f, face, detected, loaded_model)
        x += 1
        key = cv2.waitKey(20)

        # exit on esc
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def dir_classes(directory: str):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir)
    if not classes:
        raise FileNotFoundError("couldn't find classes in " + directory)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def plot_loss_curve(res: dict[str, list[float]], epochs: range):
    """
    Display the loss and accuracy curves over epochs for train and test data
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


def plot_confusion_mat(model: torch.nn.Module, test_dl: torch.utils.data.DataLoader,
                       test_data: torch.utils.data.dataset, test_targets: list):
    """
    plot the confusion matrix for a given batch of test data
    :param model: a model of class nn.Module
    :param test_dl: the dataloader of the test data
    :param test_data: the dataset of the test data
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

    confmat = ConfusionMatrix(num_classes=len(test_data.classes), task='multiclass')
    confmat_tensor = confmat(preds=pred_tensor,
                             target=torch.tensor(test_targets))

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=test_data.classes,  # turn the row and column labels into class names
        figsize=(10, 7)
    )
    plt.show()


def train(model: torch.nn.Module, train_dl: torch.utils.data.DataLoader, test_dl: torch.utils.data.DataLoader,
          l_funct: torch.nn.Module, optim: torch.optim.Optimizer, epochs: int):
    """
    Train the given model
    :param model: a model of class nn.Module
    :param train_dl: the dataloader for the train data
    :param test_dl: the dataloader for the test data
    :param l_funct: the loss function to be used
    :param optim: the optimizer to be used
    :param epochs: the number of epochs to train for
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
    train_loss, train_acc = 0, 0
    for e in range(epochs):
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(train_dl):
            X, y = X.to(device), y.to(device)
            model.train()
            y_pred = model(X)
            loss = l_funct(y_pred, y)
            train_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss /= len(train_dl)  # accumulate loss for every batch, then get avg loss per batch
        train_acc /= len(train_dl)
        res["train_loss"].append(float(train_loss.detach()))
        res["train_acc"].append(train_acc)
        test_loss, test_acc = 0, 0
        model.eval()

        with torch.inference_mode():
            for X, y in test_dl:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += l_funct(test_pred, y)
                test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
                test_acc += (test_pred_class == y).sum().item() / len(test_pred)

            test_loss /= len(test_dl)
            test_acc /= len(test_dl)
            res["test_loss"].append(float(test_loss.detach()))
            res["test_acc"].append(test_acc)

        print(e)
        print(f"\nTrain loss: {train_loss:.5f}, Train acc: {train_acc:.2f}% | Test loss: {test_loss:.5f}, "
              f"Test acc: {test_acc:.2f}%\n")

    return res


def create_model():
    image_path = Path("face_emotions")
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # show a random image
    '''
    random.seed(42)
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    img = Image.open(random_image_path)
    img.show()
    '''
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    train_transform_trivial_augment = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    '''
    full_dataset = datasets.ImageFolder(root=image_path,
                                        target_transform=None)

    classes = full_dataset.classes
    targets = full_dataset.targets

    TRAIN_SIZE = int(0.8 * len(full_dataset))
    TEST_SIZE = len(full_dataset) - TRAIN_SIZE

    train_data = torch.utils.data.random_split(full_dataset,
                                               [TRAIN_SIZE, TEST_SIZE],
                                               generator=torch.Generator().manual_seed(42))

    test_data = torch.utils.data.random_split(full_dataset,
                                              [TRAIN_SIZE, TEST_SIZE],
                                              generator=torch.Generator().manual_seed(42),
                                              trans)
    '''

    train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                      transform=data_transform,  # transforms on images
                                      target_transform=None)  # transforms on labels
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform,
                                     target_transform=None)

    print(train_data[0])

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,  # samples per batch
                                  num_workers=NUM_WORKERS,  # how much compute power used to load the data
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,  # samples per batch
                                 num_workers=NUM_WORKERS,  # how much compute power used to load the data
                                 shuffle=False)

    torch.manual_seed(12)
    TVGG_model = TinyVGG_model(in_shape=1, out_shape=len(train_data.classes), hidden_units=32, dropout_rate=0.5).to(device)
    TVGG_model.apply(init_weights)

    l_funct = nn.CrossEntropyLoss()
    epochs = 50
    optim = torch.optim.Adam(
        TVGG_model.parameters(),
        lr=0.001,  # Learning rate
        betas=(0.9, 0.999),  # Coefficients for running averages of gradient and its square
        eps=1e-8,  # Term for numerical stability
        weight_decay=0,  # L2 penalty (regularization)
        amsgrad=False  # Use AMSGrad variant
    )
    res_dict = train(TVGG_model, train_dataloader, test_dataloader, l_funct, optim, epochs)

    # plot_confusion_mat(TVGG_model, test_dataloader, test_data) MUST FIX LATER, ISSUE WITH DATASET TARGETS

    plot_loss_curve(res_dict, range(epochs))
    plot_confusion_mat(TVGG_model, test_dataloader, test_data, test_data.targets)

    model_path = Path("Models")
    model_path.mkdir(parents=True, exist_ok=True)
    save_path = model_path / "emotion_classifier_0.pth"
    torch.save(obj=TVGG_model.state_dict(), f=save_path)


def optimize_model():
    """
    Uses Bayesian optimization in order to find the best hyperparameters for the model
    """
    image_path = Path("face_emotions")
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    BATCH_SIZE = 32
    NUM_WORKERS = 0

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    # data augmentation by applying a single random transformation to each image
    train_transform_trivial_augment = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                      transform=data_transform,  # transforms on images
                                      target_transform=None)  # transforms on labels

    # create a separate dataset for augmented training data
    train_data_aug = datasets.ImageFolder(root=train_dir,  # target folder of images
                                          transform=train_transform_trivial_augment,  # transforms on images
                                          target_transform=None)  # transforms on labels

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform,
                                     target_transform=None)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,  # samples per batch
                                  num_workers=NUM_WORKERS,  # how much compute power used to load the data
                                  shuffle=True)

    train_dataloader_aug = DataLoader(dataset=train_data_aug,
                                      batch_size=BATCH_SIZE,  # samples per batch
                                      num_workers=NUM_WORKERS,  # how much compute power used to load the data
                                      shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,  # samples per batch
                                 num_workers=NUM_WORKERS,  # how much compute power used to load the data
                                 shuffle=False)

    torch.manual_seed(11)

    client = Client()
    client.configure_experiment(
        name="CNN_acc",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",  # Continuous variable
                "bounds": [1e-5, 0.1],
                "value_type": "float",
                "log_scale": True,  # Sample logarithmically "
            },
            {
                "name": "dropout_rate",
                "type": "range",  # Continuous variable
                "bounds": [1e-5, 0.1],
                "value_type": "float",
                "log_scale": True,  # Sample logarithmically "
            },


            RangeParameterConfig(
                name="learning rate",
                bounds=(0.0001, 0.01),
                parameter_type="float",
            ),
            RangeParameterConfig(
                name="dropout rate",  # probability of any unit being dropped during a training cycle
                bounds=(0.0, 0.9),
                parameter_type="float",
            ),
            RangeParameterConfig(
                name="data augmentation",  # whether to use the regular training dataset or the augmented one
                bounds=(0, 1),
                parameter_type="int",
            ),
        ],
    )

    # uses the test loss on the final epoch to
    client.configure_optimization(objective="test loss")

    for _ in range(20):
        t1 = cv2.getTickCount()
        # Use higher value of `max_trials` to run trials in parallel.
        for trial_index, parameters in client.get_next_trials(max_trials=1).items():
            TVGG_model = TinyVGG_model(in_shape=1, out_shape=len(train_data.classes),
                                       hidden_units=parameters["hidden units"],
                                       dropout_rate=parameters["dropout rate"]).to(device)
            TVGG_model.apply(init_weights)

            l_funct = nn.CrossEntropyLoss()
            epochs = 50
            optim = torch.optim.SGD(params=TVGG_model.parameters(), lr=parameters["learning rate"])
            if parameters["data augmentation"] == 0:
                res_dict = train(TVGG_model, train_dataloader, test_dataloader, l_funct, optim, epochs)
            else:
                res_dict = train(TVGG_model, train_dataloader_aug, test_dataloader, l_funct, optim, epochs)

            client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "test accuracy": res_dict["test_loss"][-1]
                },
            )
        t2 = cv2.getTickCount()
        print((t2 - t1) / cv2.getTickFrequency())

    client.get_best_parameterization()


create_model()
