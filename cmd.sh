export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# SIM train & inference
python create_tfrecord.py --data_dir=dataset-sdcnd-capstone/sim_training_data --label_map_path=./label_map.pbtxt --img_h=600 --img_w=800 --output_path=sim.record

python object_detection/legacy/train.py  \
--pipeline_config_path=trained_models/config/sim/faster_rcnn_resnet101_coco_carnd_sim.config \
--train_dir=finetuned_models_sim/faster_rcnn_resnet101_coco

python object_detection/export_inference_graph.py \
--pipeline_config_path=trained_models/config/sim/faster_rcnn_resnet101_coco_carnd_sim.config \
--trained_checkpoint_prefix=finetuned_models_sim/faster_rcnn_resnet101_coco/model.ckpt-10000 \
--output_directory=frozem_sim/faster_rcnn_resnet101_coco

# REAL train & inference
python create_tfrecord.py --data_dir=dataset-sdcnd-capstone/real_training_data --label_map_path=./label_map.pbtxt --img_h=1096 --img_w=1368 --output_path=real.record

python object_detection/legacy/train.py  \
--pipeline_config_path=trained_models/config/real/faster_rcnn_resnet101_coco_carnd_real.config \
--train_dir=finetuned_models_real/faster_rcnn_resnet101_coco