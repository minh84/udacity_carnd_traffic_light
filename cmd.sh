export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# SIM train & inference
python create_tfrecord.py --data_dir=dataset-sdcnd-capstone/sim_training_data --label_map_path=./label_map.pbtxt --img_h=600 --img_w=800 --output_path=sim.record

python object_detection/export_inference_graph.py --pipeline_config_path=trained_models/config/ssd_inception_v2_coco_carnd_sim.config --trained_checkpoint_prefix=finetuned_models/ssd_inception_v2_coco/model.ckpt-6000 --output_directory=ssd_v2_frozem_sim

# REAL train & inference
python create_tfrecord.py --data_dir=dataset-sdcnd-capstone/real_training_data --label_map_path=./label_map.pbtxt --img_h=1096 --img_w=1368 --output_path=real.record
