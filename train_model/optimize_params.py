"""
Contains functions for automating parameter search using Bayesian optimization
"""
import copy
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
import torch
from torch import nn
import torchvision
from torchvision import transforms
import data_setup
import cv2
import engine
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "smaller_dataset/train"
test_dir = "smaller_dataset/test"

experiment_path = Path("experiments")
experiment_path.mkdir(parents=True, exist_ok=True)

# get the pretrained weights and the transforms required for data to processed by the model
weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
auto_transform = weights.transforms()

# compose the model-specific transforms with a trivial data augmentation transform
data_aug_transform = transforms.Compose([
    auto_transform,
    transforms.TrivialAugmentWide(num_magnitude_bins=31)   # take a random augmentation, apply it with random strength
])

NUM_WORKERS = 0
NUM_EPOCHS = 35

LOAD_EXPERIMENT = False
PATH = "Experiments/Trial3.json"

torch.manual_seed(11)

print("INFO:    creating experiment...")

# contains the hyperparameters which are to be optimized for model performance
if LOAD_EXPERIMENT:
    client = Client.load_from_json_file(filepath=PATH)
else:
    client = Client()
    client.configure_experiment(
        name="CNN classifier",
        parameters=[
            RangeParameterConfig(
                name="learning rate",
                bounds=(0.0001, 0.01),
                parameter_type="float",
                scaling="log"
            ),
            RangeParameterConfig(
                name="L2 regularization",
                bounds=(0.0001, 0.01),
                parameter_type="float",
                scaling="log"
            ),
            RangeParameterConfig(
                name="momentum",
                bounds=(0.0001, 0.01),
                parameter_type="float",
            ),
            RangeParameterConfig(
                name="dropout rate",  # probability of any unit being dropped during a training cycle
                bounds=(0.0, 0.9),
                parameter_type="float",
            ),
            ChoiceParameterConfig(
                name="data augmentation",  # whether to use the regular training dataset or the augmented one
                values=[0],
                parameter_type="int",
                is_ordered=False
            ),
            ChoiceParameterConfig(
                name="batch size",  # whether to use the regular training dataset or the augmented one
                values=[16, 32, 64, 128, 256],
                parameter_type="int",
                is_ordered=True,
            ),
            ChoiceParameterConfig(
                name="optimizer",
                values=["Adam", "SGD", "RMSprop"],
                parameter_type="str",
                is_ordered=False,
            ),
        ],
    )

# uses the mean of the test loss over the last 5 epochs to evaluate model performance, the - indicates that this
# should be minimized
client.configure_optimization(objective="-last 5 mean test loss")

print("INFO:    loading models...")

# in order to get a fresh model every time, we keep a base model and copy the state into another model every epoch
# load base model
MNV2_base_model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=weights).to(device)
for param in MNV2_base_model.features.parameters():
    param.requires_grad = False  # freeze the feature parameters so that they are not affected by training

class_names = data_setup.get_classes(train_dir)

MNV2_base_model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=len(class_names),  # same number of output units as our number of classes
                    bias=True)).to(device)

# save the state dict for the base model
base_state_dict = MNV2_base_model.state_dict()

# load the second model which will be used for each trial
MNV2_model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=weights).to(device)
for param in MNV2_base_model.features.parameters():
    param.requires_grad = False  # freeze the feature parameters so that they are not affected by training

MNV2_model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=len(class_names),  # same number of output units as our number of classes
                    bias=True)).to(device)

for x in range(30):
    t1 = cv2.getTickCount()
    # Use higher value of `max_trials` to run trials in parallel
    for trial_index, parameters in client.get_next_trials(max_trials=1).items():

        # create dataloaders
        train_dataloader, test_dataloader, class_names, test_targets = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=auto_transform,
            test_transform=auto_transform,
            batch_size=parameters["batch size"],
            num_workers=0
        )

        # load the state of the base model every trial, uses deepcopy to
        MNV2_model.load_state_dict(copy.deepcopy(base_state_dict))

        l_funct = nn.CrossEntropyLoss()
        epochs = 50
        optim = None
        if parameters["optimizer"] == "SGD":
            optim = torch.optim.SGD(
                params=MNV2_model.parameters(),
                lr=parameters["learning rate"],
                momentum=parameters["momentum"],
                weight_decay=parameters["L2 regularization"]
            )
        elif parameters["optimizer"] == "Adam":
            optim = torch.optim.Adam(
                params=MNV2_model.parameters(),
                lr=parameters["learning rate"],
                weight_decay=parameters["L2 regularization"]
            )
        else:
            optim = torch.optim.RMSprop(
                params=MNV2_model.parameters(),
                lr=parameters["learning rate"],
                momentum=parameters["momentum"],
                weight_decay=parameters["L2 regularization"]
            )
        if parameters["data augmentation"] == 0:
            res_dict = engine.train(
                model=MNV2_model,
                train_dl=train_dataloader,
                test_dl=test_dataloader,
                l_funct=l_funct,
                optim=optim,
                epochs=NUM_EPOCHS,
                device=device
            )
        else:
            aug_train_dataloader = data_setup.get_aug_dataset(
                data_dir=train_dir,
                transform=data_aug_transform,
                batch_size=parameters["batch size"],
                num_workers=0
            )
            res_dict = engine.train(
                model=MNV2_model,
                train_dl=aug_train_dataloader,
                test_dl=test_dataloader,
                l_funct=l_funct,
                optim=optim,
                epochs=NUM_EPOCHS,
                device=device
            )

        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "last 5 mean test loss": sum(res_dict["test_loss"][-5:])/5
            },
        )

        print("INFO:    Last 5 mean test loss: " + str(sum(res_dict["test_loss"][-5:])/5))
    t2 = cv2.getTickCount()
    print("INFO:    Time for trial: " + str((t2 - t1) / cv2.getTickFrequency()))
    print("INFO:    Saving experiment state...")
    path = "experiments/Trial" + str(x) + ".json"
    client.save_to_json_file(path)

client.get_best_parameterization()
