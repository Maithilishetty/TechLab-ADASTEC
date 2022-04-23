# Traffic Light Recognition Using Deep Learning

This repository contains the code for our project carried out as part of TechLab @ MCity while working with [ADASTEC](https://www.adastec.com/). ADASTEC specializes in the delivery of level 4 automated driving software that is ready for real-world transit application. As part of our winter 2022 project, we implemented traffic light recognition using Deep Learning. In particular, we implemented Faster R-CNN and YOLOv4 to accomplish this task. 

The team members for this project are: 
- Maithili Shetty (mjshetty@umich.edu) 
- Leo Lee (leowcl@umich.edu)
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
- Edit the config file in the faster R-CNN config folder to point to the tfrecords and label map files (marked as PATHTO). Additonally, also edit the fine_tune_checkpoint path to point to the weights of the pretrained faster R-CNN inception model which was previously downloaded 
- Change directory into the donwloaded TensorFlow Object Detection folder 
- To start training, run: 
  ```
  python legacy/train.py --logtostderr --train_dir=./models/train/ --pipeline_config_path=/PATHTO/config/faster_rcnn_inception_v2_coco.config
  ```
- To generate a frozen model for evaluation, run: 
  ```
  python export_inference_graph.py --input_type image_tensor --pipeline_config_path /PATHTO/config/faster_rcnn_inception_v2_coco.config --  trained_checkpoint_prefix ./models/train/model.ckpt --output_directory models
  ```
- To evaluate and obtain the mAP values, run: 
  ```
   python eval.py --logtostderr --checkpoint_dir=/ATHTO/checkpoint_dir --eval_dir=path/to/eval_dir --pipeline_config_path=path/to/pretrained_model.config
  ``` 

## YOLOv4
One of the best thing about YOLO is that it is running out of the box and supports well known datasets such as Pascal VOC, COCO and Open Images dataset. Let's clone and make it with:

```
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
```

Now we need some example weights to run YOLO, download it from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and save it into darknet folder.

Now we can run an example:

```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

A result image will appear and we can see that YOLO found a dog, a bicycle and a truck. YOLO can be used for multiple images, with webcam and videos.

### Preparing the Dataset for Traning

As we already know that we need labels for the input images. In a standart CNN it would be a label for each image but since we are looking for parts of one image we need more than this. So YOLO is asking for a .txt file for each image as:

```
<object-class> <x> <y> <width> <height>
```

Bosch Small Traffic Lights Dataset is coming with a Python script which turns the dataset into Pascal-VOC like dataset. It is good because YOLO has a script for converting VOC dataset to YOLO styled input. First clone the repository into the extracted dataset folder:  
```
git clone https://github.com/bosch-ros-pkg/bstld.git
```

### Data Folders Preparation
**Warning:** Extracted folder has white spaces in it's name. Please avoid white spaces and replace them with '-' (i.e Bosch-Traffic-Light-Dataset). Otherwise paths will be unusable in some cases! 

To keep dataset in order we will create 3 folders under rgb/train/
1. traffic_light_images
2. traffic_light_xmls
3. traffic_light_labels

```
cd rgb/train
```

Since images are in separate folders and it will be easier to manipulate them when they are in one folder let's put them all together under rgb/train/traffic_light_images folder.

```
mkdir traffic_light_images
find . -type f -print0 | xargs -0 --no-run-if-empty cp --target-directory=traffic_light_images
```
If you do not want to waste your space you should change 'cp' with 'mv' to move the images instead of making a copy of them. Now we have all the images under traffic_light_images folder.

Now create xmls folder and run:

```
mkdir traffic_light_xmls
```
**Update:** PyYaml's load function has been <a href=https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col>deprecated</a>, so if you are getting an error with yaml.load() you should change bosch_to_pascal.py line 60 to yaml.safe_load()

Now go back to top Bosch-Traffic-Light-Dataset folder and run bosch_to_pascal.py script from bstld, which will create necessary xml files for training with YOLO. Where first argument is PATH_TO_DATASET/train.yaml and second argument is rgb/train/traffic_light_xmls folder which we recently created:
```html
cd ../..
python bstld/bosch_to_pascal.py train.yaml rgb/train/traffic_light_xmls/
```

Now we have 5093 xml label files but we have to convert VOC to YOLO type labels with the script from darknet. So create a traffic_light_labels folder to /rgb/train/

```html
mkdir rgb/train/traffic_light_labels
```

### darknet/traffic-lights folder
Let's go back to the darknet folder and create a folder named traffic-lights. We will put our files in this folder to reach them easily.

```html
mkdir traffic-lights 
```

#### VOC -> YOLO
From darknet/scripts folder, make a copy of the voc_label.py and name it bosch_voc_to_yolo_converter.py and put it under traffic-lights folder. This script will convert VOC type labels to YOLO type labels.

```html
cp scripts/voc_label.py traffic-lights/bosch_voc_to_yolo_converter.py
```

Here we have to change classes names with our class names from the dataset.

```Python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys

sets=['traffic_lights']

classes = ["RedLeft", "Red", "RedRight", "GreenLeft", "Green", "GreenRight", "Yellow", "off"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path_input,file_folder,file_name):
    in_file = open('%s'%(xml_path_input))
    out_file = open('%s/%s.txt'%(file_folder,file_name), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

output_folder = str(sys.argv[1])
xmls_list = str(sys.argv[2])
images_folder = str(sys.argv[3])

for image_set in sets:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    xml_paths = open(xmls_list).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    for xml_path in xml_paths:
        #print("xml path: ",xml_path)
        xml_name = xml_path.split('/')[-1]
        #print("xml name:",xml_name)
        image_name = xml_name.split('.')[0]
        #print("image name: ",image_name)
        #print(images_folder+'/%s.png\n'%(image_name))
        list_file.write(images_folder+'%s.png\n'%(image_name))
        convert_annotation(xml_path,output_folder,image_name)
    list_file.close()
 ```
 
 And for the arguments, we have to give:
1. output_folder for .txt files (PATH_TO_DATASET/rgb/train/traffic_light_labels)
2. xmls_list which is a .txt file that has the paths to the xml files and (we will create next)
3. images folder path which we are going to use for training. (PATH_TO_DATASET/rgb/train/traffic_light_images)

We need the paths of the .xml files as a list in a .txt file, in order to get it we will write a little Python script:

```html
cd traffic-lights
subl make_xml_list.py
```

```Python
import os
import sys

xmls_path = sys.argv[1] #xml files path

xml_files = []

#r = root, d = directories, f = xml_files

for r,d,f in os.walk(xmls_path):
	for file in f:
		if '.xml' in file:
			xml_files.append(os.path.join(r, file)) #Gets the whole file xmls_path
			#xml_files.append(os.path.splitext(file)[0]) # Gets only the name of the file without extension,path etc.	

file_num = len(xml_files)
print("Length of the .xml xml_files: ", file_num)

if not open('bosch_traffic_light_xmls_list.txt','w'):
	os.makefile('bosch_traffic_light_xmls_list.txt')

labels = open('bosch_traffic_light_xmls_list.txt','w')

for xml in xml_files:
	labels.write(xml + '\n')

labels.close()

#for f in xml_files:
	#print(f)
```

Save and run it:

```html
python make_xml_list.py PATH_TO_DATASET/rgb/train/traffic_light_xmls/
```
It will create bosch_traffic_light_xmls_list.txt file.

Let's copy the data/voc.names to traffic-lights and name it voc-bosch.names:

```html
cp ../data/voc.names voc-bosch.names
subl voc-bosch.names
```

and replace the items with:

1. RedLeft
2. Red
3. RedRight
4. GreenLeft
5. Green
6. GreenRight
7. Yellow
8. off

Now we can convert VOC to YOLO format:

We will use the folder PATH_TO_DATASET/rgb/train/traffic_light_labels for outputs, 
recently created bosch_traffic_light_xmls_list.txt and 
PATH_TO_DATASET/rgb/train/traffic_light_images for training images.

```html
python bosch_voc_to_yolo_converter.py ~/Datasets/Bosch-Traffic-Light-Dataset/rgb/train/traffic_light_labels/ bosch_traffic_light_xmls_list.txt ~/Datasets/Bosch-Traffic-Light-Dataset/rgb/train/traffic_light_images/
```

We have to create train.txt and test.txt which are list of the paths' of the relative images. Write a basic splitter script named train_test_split.py:

```html
subl train_test_split.py
```

```Python
import glob, os
import numpy as np
from sklearn.model_selection import train_test_split
import sys

image_paths_file = sys.argv[1] #traffic_lights/traffic_lights.txt

# Percentage of images to be used for the test set (float between 0-1)
percentage_test = float(sys.argv[2]);

img_paths = []
img_paths = open(image_paths_file).read().strip().split()

X_train, X_test= train_test_split(img_paths, test_size=percentage_test, random_state=31)

with open('train.txt', 'a') as train_file:
	for train in X_train:
		train_file.write(train + '\n')

with open('test.txt', 'a') as test_file:
	for test in X_test:
		test_file.write(test + '\n')
```

It takes recently created traffic_lights.txt file as first argument and second argument split percentage between 0 to 1. 

```html
python train_test_split.py traffic_lights.txt 0.2
```
test.txt and train.txt is created.

Create a backup folder inside traffic-lights folder where we will save our weights as we train:

```html
mkdir backup
```

Make a copy of the cfg/voc.data and name it voc-bosch.data .

```html
cp ../cfg/voc.data voc-bosch.data
```
Open it:

```html
classes= 20
train  = /home/pjreddie/data/voc/train.txt
valid  = /home/pjreddie/data/voc/2007_test.txt
names = data/voc.names
backup = backup
```

classes shows the number of the labels we would like to classify. From the dataset we can see that main lights are RedLeft, Red, RedRight, GreenLeft, Green, GreenRight, Yellow and off. Feel free to add or extract the ones you like. So our classes will be '8'. train.txt and test.txt are the text files which has the paths of the image files. names, are labels' names and as mentioned before we should get them from the database. Let's start updating the voc-bosch.data:

```html
classes= 8
train  = traffic-lights/train.txt
valid  = traffic-lights/test.txt
names = traffic-lights/voc-bosch.names
backup = traffic-lights/backup
```

Now we need one more thing to do to start training. Copy yolov4-custom.cfg from darknet/cfg folder into traffic-lights folder and name it yolov4-bosch.cfg .

```html
cp ../cfg/yolov4-custom.cfg yolov4-bosch.cfg
```

Open it:

<a href="https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects">Calculate number of filters: </a>

```html
filters= 3 x (5 + #ofclasses)
```

filters = 3 x (5+8) = 39

Change filters' size before '[yolo]' parameters with 39 and classes to 8 in '[yolo]' parameters .

We will use the technique called transfer learning where we use the pre-trained VOC data and just change the end of the deep-neural-network.

Download the pre-trained weights-file (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )

- change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)
- change line subdivisions to [`subdivisions=16`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
- change line max_batches to (`classes*2000`, but not less than number of training images and not less than `6000`), f.e. [`max_batches=6000`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) if you train for 3 classes
- change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L22)
- set network size `width=416 height=416` or any value multiple of 32: 

```html
cd ..
./darknet detector train traffic-lights/voc-bosch.data traffic-lights/yolov4-bosch.cfg yolov4.conv.137
```

Too generate map for validation set while training:
```html
./darknet detector train traffic-lights/voc-bosch.data traffic-lights/yolov4-bosch.cfg yolov4.conv.137 -map
```

After training done try it with:
```html
./darknet detector demo traffic-lights/voc-bosch.data traffic-lights/yolov4-bosch.cfg traffic-lights/backup/yolov4-bosch_best.weights <video file>
```

## Results 
<img src="images/demo_yolo.png" alt="demo_yolo">

<img src="images/demo_faster-rcnn.png" alt="demo_faster-rcnn">

<img src="images/chart_6000.png" alt="training loss">

<img src="images/chart_10000.png" alt="training loss + validation map">

## References:

1. <a href="https://github.com/AlexeyAB/darknet"> YOLO Github Repo </a>
2. <a href="https://hci.iwr.uni-heidelberg.de/node/6132"> BOSCH Traffic Lights Dataset</a>
3. <a href="https://github.com/bosch-ros-pkg/bstld"> Bosch Small Traffic Lights Dataset Github Repository</a>
4. <a href="https://github.com/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset"> Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset</a>
