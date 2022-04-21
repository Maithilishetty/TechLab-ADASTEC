# Traffic Light Recognition Using Deep Learning

This repository contains the code for our project carried out as part of TechLab @ MCity while working with [ADASTEC](https://www.adastec.com/). ADASTEC specializes in the delivery of level 4 automated driving software that is ready for real-world transit application. As part of our winter 2022 project, we implemented traffic light recognition using Deep Learning. In particular, we implemented Faster R-CNN and YOLOv4 to accomplish this task. 

The team members for this project are: 
- Maithili Shetty (mjshetty@umich.edu) 
- Leo Lee 
- Daphne Tsai 
- Niyanta Mehra 

## Dataset 

For training and testing our model, we used the [BOSCH Small Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset). This dataset can be downloaded from their website. 

## Faster R-CNN 

Faster R-CNN is a deep convolutional network that is used extensively for the purpose of object detection. For more details on this methods, refer to: https://arxiv.org/abs/1506.01497. For the purpose of training this model, we make use of the Tensorflow object detection API. This section lays out the general instructions to train and evaluate the model using TensorFlow. 

- Clone the [TensorFlow Object Detection API Repository](https://github.com/tensorflow/models/tree/master/research/object_detection) 
- Download the pretrained [faster_rcnn_inception_v2_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) model. 
- Download and unzip the dataset files
- Run the to_tfrecord.py script to generate tfrecords for training and testing 
  ```
  python to_tfrecord.py --train_yaml <path_to_train_yaml> --test_yaml <path_to_test_yaml> -dataset_folder <path_to_unzipped_data_folder> 
  ```
- Edit the generated tfrecord and label map files in the faster R-CNN config file (marked as PATHTO). Additonally, also edit the fine_tune_checkpoint path to point to the weights of the pretrained faster R-CNN inception model which was previously downloaded 
- Change directory into the donwloaded TensorFlow Object Detection folder 
- To start training, run: 
  ```
  python legacy/train.py --logtostderr --train_dir=./models/train/ --pipeline_config_path=/PATHTO/config/faster_rcnn_inception_v2_coco.config
  ```

