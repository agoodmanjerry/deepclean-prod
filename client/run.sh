#! /bin/bash -e

BATCH_SIZE=$1
TAG=20.07

cd ..
docker build -t $USER/deepclean:client -f client/Dockerfile --build-arg tag=$TAG.

docker run --rm -it -v $PWD:/workspace --gpus 1 -u $(id -u):$(id -g) \
    $USER/deepclean:client python client/export_model.py \
        --chanslist example_configs/chanslist.ini \
        --batch-size $BATCH_SIZE \
        --model-store-dir ./modelstore

docker run --rm -d -v $PWD/modelstore:/modelstore --gpus 1 -u $(id -u):$(id -g) \
    --name tritonserver nvcr.io/nvidia/tritonserver:$TAG-py3 bin/tritonserver \
        --model-store /modelstore --model-control-mode=explicit

# TODO: add args for clean duration and model name
docker run --rm -it -v $PWD:/workspace --gpus 0 -u $(id -u):$(id -g) \
    $USER/deepclean:client python client/measure_throughput.py \
        --chanslist example_configs/chanslist.ini \
        --clean-duration 3600 \
        --batch-size $BATCH_SIZE \
        --model-name deepclean_trt_fp16 \
        --model-version 1
