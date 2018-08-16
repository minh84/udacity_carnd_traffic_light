#!/bin/bash
TRAIN_TYPE=$1
set -x

python object_detection/legacy/train.py \
--pipeline_config_path=config/$TRAIN_TYPE/ssd_mobilenet_v1_coco_carnd.config \
--train_dir=finetuned_${TRAIN_TYPE}/ssd_mobilenet_v1_coco

python object_detection/legacy/train.py \
--pipeline_config_path=config/$TRAIN_TYPE/ssd_inception_v2_coco_carnd.config \
--train_dir=finetuned_${TRAIN_TYPE}/ssd_inception_v2_coco

python object_detection/legacy/train.py \
--pipeline_config_path=config/$TRAIN_TYPE/faster_rcnn_resnet101_coco_carnd_sim.config \
--train_dir=finetuned_${TRAIN_TYPE}/faster_rcnn_resnet101_coco
