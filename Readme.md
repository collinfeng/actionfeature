### Docker Image Buidling Command: 
docker build --build-arg UID=26 -t cf-actionfeature .

### Docker Command to create and auto-delete Container(from image) at Run-Time on Host
docker run --rm --name collin-runtime -v $(pwd):/home/ActionFeature -w /home/ActionFeature -u $(id -u):$(id -g) -it --gpus "device=2" cf-actionfeature python3 main.py

docker run --rm --name collin-runtime -v $(pwd):/home/ActionFeature -w /home/ActionFeature -u $(id -u):$(id -g) -it cf-actionfeature python3 main.py

OR Instead execute cmd+P dev container open folder 
