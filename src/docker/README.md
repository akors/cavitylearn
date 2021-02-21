
# Docker images for cavitylearn

This image was built for archival purposes and can be used to reproduce previous cavitylearn results.

## Building docker image

The paths in the dockerfile are relative to the /src/ directory of the repository. The docker image can be built like
this:

```
docker build -f docker/Dockerfile  . \
  -t registry.innophore.com/cavitylearn:latest -t registry.innophore.com/cavitylearn:1.0 \
  --label org.opencontainers.image.created=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --label org.opencontainers.image.source="https://git.innophore.com/akors/cavitylearn" \
  --label org.opencontainers.image.revision=$(git rev-parse HEAD)
```

To push this image to the innophore registry, run:

```
docker push registry.innophore.com/cavitylearn:latest registry.innophore.com/cavitylearn:1.0
```

## Running

To run the code in this container image with NVIDIA GPU acceleration, the NVIDIA Container Toolkit is required. 
Refer to [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/index.html) on how to install, configure and use it.

If the toolkit is installed and you are using the docker container runtime, you can select the GPU that should be used
by passing its index to docker via the `--gpus=0` parameter. Notice that only the first GPU will be used by cavitylearn.

### Training

If the decompressed dataset is found in `./data/shaped-commonligands02` directory, you can start the training like this:

```
docker run --gpus=0 -u $(id -u):$(id -g) -it -v $(pwd)/data:/data/ registry.innophore.com/cavitylearn:latest \
  cavtrainer.py /data/shaped-commonligands02 /data/shaped-commonligands02_result \
    --name fromdocker --track-accuracy \
    --batches=2000 --batchsize=100 \
    --learnrate=1e-3 --learnrate-decay=0.6813 --learnrate-decay-frequency=1000 \
    --lambda=0.1 --keepprob-conv=.95 --keepprob-fc=.90
```

### Tensorboard

The `tensorboard` utility can be used to monitor the progress of the training. If the output of the training was written
to `./data/shaped-commonligands02_result`, you can start tensorboard like this:

```
docker run -u $(id -u):$(id -g) -v $(pwd)/data:/data/ -p 6006:6006 registry.innophore.com/cavitylearn:latest \
  tensorboard --logdir=/data/shaped-commonligands02_result/logs
```

You can connect to the tensorboard via the following URL: http://localhost:6006/ .

### Jupyter

```
docker run --gpus=0 -u $(id -u):$(id -g) -v $(pwd)/data:/data/ -p 8888:8888 registry.innophore.com/cavitylearn:latest \
  jupyter notebook --allow-root --no-browser --ip=0.0.0.0
```

You can connect to jupyter via the following URL: http://localhost:8888/ .
