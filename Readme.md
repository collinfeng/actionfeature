## Introduction 
This is an active repo for action feature project, include the JAX implementation of (https://arxiv.org/abs/2201.12658) and other in-developing experimental code

## To Run the code:
1. modify the config dict as needed
2. make sure you have docker installed

### Example Docker Image Buidling Command: 
docker build --build-arg UID=26 -t cf-actionfeature .

### Example Docker Command to create and auto-delete Container(from image) at Run-Time on Host
docker run --rm --name collin-runtime -v $(pwd):/home/ActionFeature -w /home/ActionFeature -u $(id -u):$(id -g) -d -it --gpus "device=2" cf-actionfeature python3 main.py

<!-- docker run --rm --name collin-runtime -v $(pwd):/home/ActionFeature -w /home/ActionFeature -u $(id -u):$(id -g) -it cf-actionfeature python3 main.py -->

### Use cmd+P dev container reopen in VSC for interactive sessions
