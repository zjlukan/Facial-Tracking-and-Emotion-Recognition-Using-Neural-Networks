"""
Trains an image classification pytorch model
"""
import torch
from torch import nn
import data_setup, engine, utils
import torchvision
import visualize_results
import cv2
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=50)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--dropout", default=0.2)
parser.add_argument("--momentum", default=0.001)
parser.add_argument("--L2_reg", default=0.001)
parser.add_argument("--optimizer", default="Adam")
parser.add_argument("--train_dir", default="train")
parser.add_argument("--test dir", default="test")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(12)

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
MOMENTUM = args.momentum
L2_REG = args.L2_reg

t1 = cv2.getTickCount()

train_dir = args.train_dir
test_dir = args.test_dir

# get the pretrained weights and the transforms required for data to processed by the model
weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
auto_transform = weights.transforms()

print("INFO:    creating dataloaders...")

# create the train and test dataloaders and get the class names and targets of the test dataset
train_dataloader, test_dataloader, class_names, test_targets = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=auto_transform,
    test_transform=auto_transform,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# download the MobileNet v2 model from pytorch hub and freeze the parameters in the "features" section
MNV2_model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=weights).to(device)
for param in MNV2_model.features.parameters():
    param.requires_grad = False

# set the hyperparameters for the model and change the number of output features to fit the data
MNV2_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=args.dropout, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=len(class_names),  # same number of output units as our number of classes
                        bias=True)).to(device)

# use cross entropy loss for classification task
l_funct = nn.CrossEntropyLoss()

# set the hyperparameters for the optimizer
optim = torch.optim.Adam(
        MNV2_model.parameters(),
        lr=0.001,  # Learning rate
        betas=(0.9, 0.999),  # Coefficients for running averages of gradient and its square
        eps=1e-8,  # Term for numerical stability
        weight_decay=0,  # L2 penalty (regularization)
        amsgrad=False,  # Use AMSGrad variant
    )

if args.optimizer == "SGD":  # use Stochastic Gradient Descent optimizer
    optim = torch.optim.SGD(
        params=MNV2_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=L2_REG
    )
elif args.optimizer == "Adam":  # use Adam optimizer
    optim = torch.optim.Adam(
        params=MNV2_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=L2_REG
    )
elif args.optimizer == "RMSprop":
    optim = torch.optim.RMSprop(  # use Root Mean Square Propagation optimizer
        params=MNV2_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=L2_REG
    )
else:
    raise ValueError("optimizer argument must be one of: 'Adam', 'RMSprop', or 'SGD'")

print("INFO:    Model created successfully, starting training...")

# train the model
res_dict = engine.train(
    model=MNV2_model,
    train_dl=train_dataloader,
    test_dl=test_dataloader,
    l_funct=l_funct,
    optim=optim,
    epochs=NUM_EPOCHS,
    device=device
)

print("INFO:    Visualizing results...")

# display a loss curve and confusion matrix
visualize_results.plot_loss_curve(res_dict, range(NUM_EPOCHS))
visualize_results.plot_confusion_mat(MNV2_model, test_dataloader, class_names, test_targets)

print("INFO:    Saving model...")

# save the model to file
utils.save_model(MNV2_model, "MNV2_classifier.pth", "Models")

print("Model saved successfully")

t2 = cv2.getTickCount()
print((t2-t1)/cv2.getTickFrequency())

