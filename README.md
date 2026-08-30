# Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks
Takes feed from camera and uses a real-time tracking system to isolate the face and feed it into a CNN

# Prerequisites    
Python: 3.6+ 
uv: Python package manager (required for installing dependencies)  

# Installation  
Use the following command in a terminal to clone the sample application repository:  
```
git clone https://github.com/zjlukan/Facial-Tracking-and-Emotion-Recognition-Using-Neural-Networks
```

Installing dependencies:  
```
uv pip install -r requirements.txt
```

# Running via a Docker image:  
Download the image on docker hub:  

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

