FROM nvcr.io/nvidia/tritonserver:20.12-py3

USER root

RUN apt-get update

RUN apt-get -y install wget

# copy yours models to models's folder
COPY server/docs/examples/model_repository/ /models/
