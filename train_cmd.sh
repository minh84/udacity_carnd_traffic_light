#!/bin/bash

python object_detection/legacy/train.py \
--pipeline_config_path=config/sim/ssd_mobilenet_v1_coco_carnd_sim.config \
--train_dir=finetuned_sim/ssd_mobilenet_v1_coco

python object_detection/legacy/train.py \
--pipeline_config_path=config/sim/ssd_inception_v2_coco_carnd_sim.config \
--train_dir=finetuned_sim/ssd_inception_v2_coco

python object_detection/legacy/train.py \
--pipeline_config_path=config/sim/faster_rcnn_resnet101_coco_carnd_sim.config \
--train_dir=finetuned_sim/faster_rcnn_resnet101_coco
