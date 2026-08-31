# Features  
**Main app:**  
* Takes feed from camera and uses a real-time tracking system to isolate the face and feed it into a CNN for classification  
* Runs locally in the browser using NiceGUI interface

**Training module:**  
* Uses transfer learning on the pre-trained MobileNetV2 model:  
>https://pytorch.org/hub/pytorch_vision_mobilenet_v2/  
* Visualizes results using a loss curve and confusion matrix  
* Device-agnostic code (utilizes Cuda if available)  

**Parameter optimization:**  
* Uses Bayesian optimization to automatically search for the optimal parameters (learning rate, dropout rate, etc) for the model evaluating it a minimal amount of times  
* Saves the state of the experiment every trial to a .json file, which can be loaded into optimize_params.py for continued searching  
* Uses the mean test loss over the last 5 trials as evaluation metric  
* Utilizes trial-level early stopping to terminate the training of unpromising models  
* It is recommended to use a smaller dataset for parameter search in order to save time before applying these parameters to the final model trained on the full dataset  

# Prerequisites    
Python: 3.6+  
uv: Python package manager (required for installing dependencies)  

# App installation  
Use the following command in a terminal to clone the app:  
```
git clone https://github.com/zjlukan/Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks
```

Installing dependencies:  
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

---

# Running the training module   
Use the following command in a terminal to clone the app: 
```
git clone https://github.com/zjlukan/Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks
```

Installing dependencies:  
```
uv pip install -r requirements-train.txt
```

# Running the training module from command line  
To train the model, run the following command:  
```
python3 ./train.py num-epochs batch-size learning-rate train-dir test-dir
```

To run Bayesian optimization parameter search, run the following command:  
```
python3 ./optimize_params num-epochs num-trials load-experiment experiment-path
```
