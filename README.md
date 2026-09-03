# Features  
**Main app:**  
* Takes feed from camera and uses a real-time tracking system to isolate the face and feed it into a CNN for classification  
* Runs locally in the browser using FastAPI with a NiceGUI interface

**Training module:**  
* Uses PyTorch to preform transfer learning on the pre-trained MobileNetV2 model:  
>https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
* All of the parameters in the "features" section are gradient frozen
* Visualizes results using a loss curve and confusion matrix  
* Device-agnostic code (utilizes Cuda if available)  

**Parameter optimization:**  
* Uses Ax to implement Bayesian optimization  
* Automatically searches for the optimal parameters (learning rate, dropout rate, etc) for the model evaluating it a minimal amount of times  
* Saves the state of the experiment every trial to a .json file, which can be loaded into optimize_params.py for continued searching  
* Uses the mean test loss over the last 5 trials as evaluation metric  
* Utilizes trial-level early stopping to terminate the training of unpromising models    

# Prerequisites    
Python: 3.6+  
uv: Python package manager (required for installing dependencies)  

# App installation  
Use the following command in a terminal to clone the repo:  
```
git clone https://github.com/zjlukan/Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks
```

Go to the **app** directory, then run the following command:  
```
uv pip install -r requirements.txt
```

# Running the app from command line  
Run the following command:
```
python3 ./main.py
```

# Running the app via a Docker image  
Download the image on docker hub:  
> link

Or run the Dockerfile to create an image: 
```
docker build -t *YOUR_DOCKER_USERNAME*/emotion-detection-image .
```

As an example, if your username is TinkaiZ, you would run the command:
```
docker build -t TinkaiZ/emotion-detection-image .
```

Once the build has completed, you can view the image by using the following command:
```
docker image ls
```

To run the image once it has been downloaded:
```
docker run -d -p 8080:8080 --name nicegui comeback77/emotion-detection-image:latest 
```

---

# Running the training module   
Use the following command in a terminal to clone the repo: 
```
git clone https://github.com/zjlukan/Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks
```

Go to the **train_model** directory, then run the following command:  
```
uv pip install -r requirements_train.txt
```

# Running the training module from command line  
To train the model, run the following command:  
```
python3 ./train.py [--num-epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--dropout DROPOUT] [--momentum MOMENTUM] [--L2_reg L2_REG] [--optimizer OPTIMIZER] [--train_dir TRAIN_DIR] [--test_dir TEST_DIR]
```
* num_epochs: the number of epoch that the model will train for, 50 by default
* batch_size: the number of samples per gradient update, 32 by default
* lr: learning rate, controls the step size during optimization, 0.001 by default
* dropout: dropout rate for regularization (fraction of neurons to drop), 0.2 by default
* momentum: momentum factor for optimizers like SGD, 0.001 by default
* L2_reg: L2 regularization strength (weight decay coefficient), 0.001 by default
* optimizer: optimization algorithm to use (e.g., 'Adam', 'SGD', 'RMSprop'), 'Adam' by default
* train_dir: the path to the directory containing training data, 'train' by default
* test_dir: the path to the directory containing testing data, 'test' by default

To run Bayesian optimization parameter search, run the following command:  
```
python3 ./optimize_params [--num-epochs NUM_EPOCHS] [--num-trials NUM_TRIALS] [--load] [--load_path LOAD_PATH] [--train_dir TRAIN_DIR] [--test_dir TEST_DIR]
```
* num_epochs: the number of epoch that the model will train for each trial, 30 by default  
* num_trials: the number of iterations to run the parameter optimization, 5 by default  
* load: use this to indicate that you want to load an experiment from file  
* load_path: the path to a .json file containing the experiment state to be loaded  
* train_dir: the path to the directory containing training data  
* test_dir: the path to the directory containing testing data  

It is recommended to use a smaller dataset for parameter search in order to save time before applying these parameters to the final model trained on the full dataset
