"""
Trains an image classification pytorch model
"""
import torch
from torch import nn
import data_setup, engine, utils
import torchvision
import visualize_results
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(12)

'''
if len(sys.argv) < 6:
    raise TypeError("Not enough arguments!")

NUM_EPOCHS = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
HIDDEN_UNITS = int(sys.argv[3])
LEARNING_RATE = float(sys.argv[4])

data_path = Path(sys.argv[5])
image_path = data_path / "face_emotions"

if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download data
    with open(data_path / "face_emotions.zip", "wb") as f:
        request = requests.get("https://github.com/zjlukan/Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks/blob/main/data/face_emotions_20%25.zip")
        print("Downloading face emotions data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "face_emotions.zip", "r") as zip_ref:
        print("Unzipping face emotions data...")
        zip_ref.extractall(image_path)

    # Remove .zip file
    os.remove(data_path / "face_emotions.zip")
    
'''

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

t1 = cv2.getTickCount()

train_dir = "smaller_dataset/train"
test_dir = "smaller_dataset/test"

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
        torch.nn.Dropout(p=0.2, inplace=True),
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
        amsgrad=False  # Use AMSGrad variant
    )

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

