### Docker Image Buidling Command: 
docker build --build-arg UID=26 -t cf-actionfeature .

### Docker Command to create and auto-delete Container(from image) at Run-Time on Host
docker run --rm --name collin-runtime -v $(pwd):/home/ActionFeature -w /home/ActionFeature -u $(id -u):$(id -g) -d -it --gpus "device=2" cf-actionfeature python3 job_dispatch.py

docker run --rm --name collin-runtime -v $(pwd):/home/ActionFeature -w /home/ActionFeature -u $(id -u):$(id -g) -it cf-actionfeature python3 job_dispatch.py

### Use cmd+P dev container reopen in VSC for interactive sessions
