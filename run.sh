#docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models
#docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/server/docs/examples/model_repository:/models tarsioonofrio/triton-serve-test  tritonserver --model-repository=/models
docker kill $(docker ps -q)
docker rm $(docker ps -a -q)
docker run --rm \
        --cpus 2 \
        -it \
        --name tritonserver \
        -p8000:8000 -p8001:8001 -p8002:8002 \
        --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
        tarsioonofrio/triton-serve-test \
        tritonserver --model-repository=/models
#        -v/server/docs/examples/model_repository:/models \
#      	--gpus all \
