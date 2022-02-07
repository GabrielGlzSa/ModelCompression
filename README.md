# ModelCompression

For building docker file.

$ docker build -f ./dockerfiles/gpu.Dockerfile -t tf .


For running the docker container with gpus and mapping the local adress of the container so that it can be accessed in a browser.

$ sudo docker run --gpus all -u $(id -u):$(id -g) -v $(pwd):/my-devel -it -p 0.0.0.0:6006:6006 tf

For running tensorboard inside the container. Run after the command from above.

$ tensorboard --logdir=./data/logs/ --host 0.0.0.0
