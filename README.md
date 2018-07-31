# Traffic Light Detection and Classification
This repository is a part of the Carnd Capstone Project, we want to train a model to detect traffic light and classify the state of the traffic light in realtime.

## Proposed Solution
This problem can be framed as an Object Detection problem, we can have four labels: Red, Yellow, Green, NoLight (no traffic-light). To solve this Object Detection problem, we can use the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). The steps are

1. Gather data: extract images from simulator or ros bags
2. Label and annotate the images and create TFRecord file (better supporting by [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection))
3. Transfer learning using trained model (can download from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)). We fine tune the trained model's weight using our own dataset (from step 2). 
4. Export our trained model for inference and test it on test-data (as described [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md))

In the following we describe in details each of the above steps

## Gather data
