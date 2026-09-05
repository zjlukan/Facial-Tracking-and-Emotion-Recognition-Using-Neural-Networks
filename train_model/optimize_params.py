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
import visualize_results
import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=30)
parser.add_argument("--num_trials", default=5)
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument("--load_path", default="experiments/Trial1.json")
parser.add_argument("--train_dir", default="smaller_dataset/train")
parser.add_argument("--test_dir", default="smaller_dataset/test")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = args.train_dir
test_dir = args.test_dir

# create experiments directory if it doesn't already exist
experiment_path = Path("experiments")
experiment_path.mkdir(parents=True, exist_ok=True)

# get the pretrained weights and the transforms required for data to processed by the model
weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
auto_transform = weights.transforms()

# compose the model-specific transforms with a trivial data augmentation transform
data_aug_transform = transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=31),  # take a random augmentation, apply it with random strength
    auto_transform
])

NUM_WORKERS = 0
NUM_EPOCHS = args.num_epochs  # should be a number divisible by 10, otherwise it'll be rounded
NUM_TRIALS = args.num_trials

LOAD_EXPERIMENT = args.load
PATH = args.load_path

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
                values=[0, 1],
                parameter_type="int",
                is_ordered=False
            ),
            ChoiceParameterConfig(
                name="batch size",  # whether to use the regular training dataset or the augmented one
                values=[8, 16, 32],
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

evaluations = []
param_names = []
params = []

for x in range(NUM_TRIALS):
    t1 = cv2.getTickCount()
    # Use higher value of `max_trials` to run trials in parallel
    for trial_index, parameters in client.get_next_trials(max_trials=1).items():
        if x == 0 and trial_index == 0:
            param_names = list(parameters.keys())
            for p in range(len(param_names)):
                params.append([])

        # params[i][j] contains the value of the ith parameter for the jth trial
        for p in range(len(param_names)):
            params[p].append(parameters[param_names[p]])

        # create dataloaders, need to do this every trial because the batch size may vary
        train_dataloader, test_dataloader, class_names, test_targets = data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=auto_transform,
            test_transform=auto_transform,
            batch_size=parameters["batch size"],
            num_workers=0
        )

        # load the state of the base model every trial, uses deepcopy to avoid modifying the state dict
        MNV2_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=parameters["dropout rate"], inplace=True),
            torch.nn.Linear(in_features=1280,
                            out_features=len(class_names),  # same number of output units as our number of classes
                            bias=True)).to(device)
        MNV2_model.load_state_dict(copy.deepcopy(base_state_dict))

        l_funct = nn.CrossEntropyLoss()
        epochs = 50
        optim = None

        # apply parameters
        if parameters["optimizer"] == "SGD":  # use Stochastic Gradient Descent optimizer
            optim = torch.optim.SGD(
                params=MNV2_model.parameters(),
                lr=parameters["learning rate"],
                momentum=parameters["momentum"],
                weight_decay=parameters["L2 regularization"]
            )
        elif parameters["optimizer"] == "Adam":  # use Adam optimizer
            optim = torch.optim.Adam(
                params=MNV2_model.parameters(),
                lr=parameters["learning rate"],
                weight_decay=parameters["L2 regularization"]
            )
        else:
            optim = torch.optim.RMSprop(  # use Root Mean Square Propagation optimizer
                params=MNV2_model.parameters(),
                lr=parameters["learning rate"],
                momentum=parameters["momentum"],
                weight_decay=parameters["L2 regularization"]
            )
        if parameters["data augmentation"] == 1:
            train_dataloader = data_setup.get_aug_dataset(
                data_dir=train_dir,
                transform=data_aug_transform,
                batch_size=parameters["batch size"],
                num_workers=0
            )

        # every 10 epochs, determine if the trial should be terminated early due to poor performance
        for t in range(0, int(NUM_EPOCHS/10)):
            print("INFO:    Progression " + str(t))
            res_dict = engine.train(
                model=MNV2_model,
                train_dl=train_dataloader,
                test_dl=test_dataloader,
                l_funct=l_funct,
                optim=optim,
                epochs=10,
                device=device
            )

            # calculate the last mean test loss of the last 3 epochs and use to evaluate model for possible early stop
            loss = sum(res_dict["test_loss"][-3:])/3
            client.attach_data(
                trial_index=trial_index,
                raw_data={"last 5 mean test loss": loss},
                progression=t,)

            # if the trial is underperforming, stop it
            if client.should_stop_trial_early(trial_index=trial_index):
                client.mark_trial_early_stopped(trial_index=trial_index)
                evaluations.append(0)
                print("INFO:    Trial stopped early")
                break

        # calculate the mean test loss of the last 5 epochs for evaluation
        final_loss = loss = sum(res_dict["test_loss"][-5:])/5
        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "last 5 mean test loss": final_loss
            },
        )

        evaluations.append(final_loss)
        print("INFO:    Trial completed")
        print("INFO:    Last 5 mean test loss: " + str(loss))

    t2 = cv2.getTickCount()
    print("INFO:    Time for trial: " + str((t2 - t1) / cv2.getTickFrequency()))
    print("INFO:    Saving experiment state...")
    path = "experiments/Trial" + str(x) + ".json"
    client.save_to_json_file(path)

best_params, pred, idx, name = client.get_best_parameterization()
print("INFO:    Trial index: " + str(idx) + " | Best parameters: " + str(best_params))

# plot the loss over each trial, as well as the individual parameters over each trial
visualize_results.plot_optim(
    evaluations=evaluations, params=params, param_names=param_names, trials=range(NUM_TRIALS))
