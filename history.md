docker pull nvcr.io/nvidia/tritonserver:20.12-py3

[comment]: <> (docker pull nvcr.io/nvidia/tritonserver:20.09-py3)

docker pull nvcr.io/nvidia/tritonserver:20.09-py3-sdk

docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $HOME/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models

[comment]: <> (docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $HOME/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:20.09-py3 tritonserver --model-repository=/models)

docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:20.12-py3-sdk
