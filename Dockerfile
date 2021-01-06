FROM nvcr.io/nvidia/tritonserver:20.12-py3

USER root

RUN apt-get update

RUN apt-get -y install curl

COPY serve/examples/ /home/model-server/examples/

RUN curl -o /home/model-server/examples/image_classifier/densenet161-8d451a50.pth https://download.pytorch.org/models/densenet161-8d451a50.pth
# RUN curl -o /home/model-server/examples/object_detector/fast-rcnn/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

USER model-server

RUN torch-model-archiver --model-name densenet161 --version 1.0 --model-file /home/model-server/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/examples/image_classifier/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /home/model-server/examples/image_classifier/index_to_name.json --handler image_classifier
# RUN torch-model-archiver --model-name fastrcnn --version 1.0 --model-file /home/model-server/examples/object_detector/fast-rcnn/model.py --serialized-file /home/model-server/examples/object_detector/fast-rcnn/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --export-path /home/model-server/model-store  --extra-files /home/model-server/examples/object_detector/index_to_name.json --handler object_detector

