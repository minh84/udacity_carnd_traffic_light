import yaml
import glob
import os
import sys
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def dict_to_tf_example(annotated_sample,
                       img_h,
                       img_w,
                       data_dir,
                       label_map_dict):
    height = img_h # Image height
    width = img_w # Image width

    filename = annotated_sample['filename'] # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(os.path.join(data_dir, filename), 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'jpg'.encode()

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in annotated_sample['annotations']:
        # if box['occluded'] is False:
        # print("adding box")
        xmins.append(float(box['xmin'] / width))
        xmaxs.append(float((box['xmin'] + box['x_width']) / width))
        ymins.append(float(box['ymin'] / height))
        ymaxs.append(float((box['ymin' ]+ box['y_height']) / height))
        classes_text.append(box['class'].encode())
        classes.append(int(label_map_dict[box['class']]))

    # convert to string
    filename = filename.encode('utf-8')

    # create tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(annotated_samples, img_h, img_w, data_dir, label_map_dict, out_file):
    writer = tf.python_io.TFRecordWriter(out_file)
    nb_samples = len(annotated_samples)
    for i, annotated_sample in enumerate(annotated_samples):
        tf_example = dict_to_tf_example(annotated_sample,
                                        img_h,
                                        img_w,
                                        data_dir,
                                        label_map_dict)
        writer.write(tf_example.SerializeToString())
        if i % 10 == 0:
            sys.stdout.write('\rPercent done {:.2f}%'.format((i * 100 / nb_samples)))

    writer.close()

    print('\nDone, TFRecord is saved to {}'.format(out_file))

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to dataset.')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('img_h', '', 'Image height')
flags.DEFINE_string('img_w', '', 'Image width')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def main(_):
    data_dir = FLAGS.data_dir
    yaml_file = glob.glob(os.path.join(data_dir, '*.yaml'))[0]
    with open(yaml_file, 'rb') as f:
        annotated_samples = yaml.load(f.read())

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    create_tf_record(annotated_samples,
                     int(FLAGS.img_h),
                     int(FLAGS.img_w),
                     data_dir,
                     label_map_dict,
                     FLAGS.output_path)

if __name__ == '__main__':
    '''
    How to use it e.g 
    python create_tfrecord.py --data_dir=dataset-sdcnd-capstone/sim_training_data \
    --label_map_path=./label_map.pbtxt --img_h=600 --img_w=800 \
    --output_path=sim.record
    
    python create_tfrecord.py --data_dir=dataset-sdcnd-capstone/real_training_data \
    --label_map_path=./label_map.pbtxt --img_h=600 --img_w=800 \
    --output_path=real.record
    '''
    tf.app.run()