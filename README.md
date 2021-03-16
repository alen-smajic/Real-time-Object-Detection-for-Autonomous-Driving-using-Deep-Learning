# Real-time Object Detection for Autonomous Driving using Deep Learning, Goethe University Frankfurt (Fall 2020)

## General Information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Instructors:**
* [Prof. Dr. Gemma Roig](http://www.cvai.cs.uni-frankfurt.de/team.html), email: roig@cs.uni-frankfurt.de
* Dr. Iuliia Pliushch
* Kshitij Dwivedi
* Matthias Fulde

**Institutions:**
  * [Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)
  * [Computational Vision & Artificial Intelligence](http://www.cvai.cs.uni-frankfurt.de/index.html)

**Project team (A-Z):**
* Duy Anh Tran
* Pascal Fischer
* Alen Smajic
* Yujin So

## Publications ##
 * [YouTube Video (YOLO)](https://www.youtube.com/watch?v=ANQczqZwaY4)
 * [YouTube Video (Faster R-CNN)](https://www.youtube.com/watch?v=3FvUOaxWnmg)
 * [ResearchGate](https://www.researchgate.net/publication/350090136_Real-time_Object_Detection_for_Autonomous_Driving_using_Deep_Learning)

## Project Description ##
<img align="left" width="390" height="" src="Result%20images%20and%20videos/Video%20Thumbnails/YOLO.png">
<img align="right" width="390" height="" src="Result%20images%20and%20videos/Video%20Thumbnails/Faster%20R-CNN.png">

<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/><br/>

Datasets drive vision progress, yet existing driving datasets are limited in terms of visual content, scene variation, the richness of annotations, and the geographic distribution and supported tasks to study multitask learning for autonomous driving.
In 2018 Yu et al. released BDD100K, the largest driving video dataset with 100K videos and 10 tasks to evaluate the progress of image recognition algorithms on autonomous driving. The dataset possesses geographic, environmental, and weather diversity, which is useful for training models that are less likely to be surprised by new conditions. Provided are bounding box annotations of 13 categories for each of the reference frames of 100K videos and 2D bounding boxes annotated on 100.000 images for "other vehicle", "pedestrian", "traffic light", "traffic sign", "truck", "train", "other person", "bus", "car", "rider", "motorcycle", "bicycle", "trailer".

The goal of our project is to detect and classify traffic objects in a video in real-time using two approaches. We trained the two state-of-the-art models YOLO and Faster R-CNN on the Berkeley DeepDrive dataset to compare their performances and achieve a comparable mAP to the current state-of-the-art on BDD100K, which is 45.7 using a hybrid incremental net. We will focus on the context of autonomous driving and compare the models performances on a live video measuring FPS and mAP.

### Dataset ###

You can download the dataset [here](https://bdd-data.berkeley.edu/)

### Model Weights ###

You can download the model weights for YOLO and Faster R-CNN at the following [link](https://drive.google.com/drive/folders/1NGOnVfMcpzedTR0NurP05FXd8zxsF9JI?usp=sharing)

To use our predefined parameters for Faster R-CNN please make sure to download each model weight file and configure it in the following way:

    Faster R-CNN\training\ckpt-26.data-00000-of-00001
    Faster R-CNN\models\inference_graph\checkpoint\ckpt-0.data-00000-of-00001
    Faster R-CNN\models\inference_graph\saved_model\variables\variables.data-00000-of-00001


### YOLO ###

The whole YOLO algorithm was implemented completely from scratch.

#### Training YOLO #####

To train the model use the script:
* ```train.py```

To execute the script you need to specify the following parameters:
* ```--train_img_files_path``` ```-tip``` (default: bdd100k/images/100k/train/) path to the train image folder
* ```--train_target_files_path``` ```-ttp``` (default: bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json) path to json file containing the train labels
* ```--learning_rate``` ```-lr``` (default: 1e-5) learning rate
* ```--batch_size``` ```-bs``` (default: 10) batch size
* ```--number_epochs``` ```-ne``` (default: 100) amount of epochs
* ```--load_size``` ```-ls``` (default: 1000) amount of batches which are being loaded in one take
* ```--number_boxes``` ```-nb``` (default: 2) amount of bounding boxes which should be predicted
* ```--lambda_coord``` ```-lc``` (default: 5) hyperparameter penalizeing predicted bounding boxes in the loss function
* ```--lambda_noobj``` ```-ln``` (default: 0.5) hyperparameter penalizeing prediction confidence scores in the loss function
* ```--load_model``` ```-lm``` (default: 1) 1 if the model weights should be loaded, else 0
* ```--load_model_file``` ```-lmf``` (default: "YOLO_bdd100k.pt") name of the file containing the model weights

Note if you want to change the name of the class labels, you need to specify this inside the code in train.py

An example execution for training would be:

    python3 train.py -tip bdd100k/images/100k/train/ -ttp bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json -lr 1e-5 -bs 10 -ne 100 -ls 1000 -nb 2 -lc 5 -ln 0.5 -lm 1 -lmf YOLO_bdd100k.pt 

#### Detecting Bounding Boxes with YOLO ####

We provide 2 scripts to detect bounding boxes on images and videos:
* ```YOLO_to_video.py```
* ```YOLO_to_image.py```

To execute the script you need to specify the following parameters:
* ```--weights``` ```-w``` path to the model weights 
* ```--threshold``` ```-t``` (default: 0.5) threshold for the confidence score of the bounding box prediction
* ```--split_size``` ```-ss``` (default: 14) size of the grid which is applied to the image for predicting bounding boxes
* ```--num_boxes``` ```-nb``` (default: 2) amount of bounding boxes which are being predicted per grid cell
* ```--num_classes``` ```-nc``` (default: 13) amount of classes which are being predicted
* ```--input``` ```-i``` path to your input image/video 
* ```--output``` ```-o``` path to your output image/video (in case of video include .mp4 at the end)

An example execution for images would be:

    python3 YOLO_to_image.py -w /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/YOLO_bdd100k.pt -t 0.5 -ss 14 -nb 2 -nc 13 -i /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/test.jpg -o /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/output.png

An example execution for videos would be:

    python3 YOLO_to_video.py -w /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/YOLO_bdd100k.pt -t 0.5 -ss 14 -nb 2 -nc 13 -i /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/test.mov -o /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/output.mp4
    
### Faster R-CNN ###

To use Faster R-CNN you have to configure the Tensorflow 2 object detection API.
You can follow this [tutorial](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model) which shows how to train, validate and export a custom object detection model.

#### Prediction #####

To predict images/videos from a specific folder, you can use the script: ```detect_objects.py``` 

An example execution for making predictions on images would be:

    python3 detect_objects.py --model_path models/inference_graph/saved_model --path_to_labelmap models/label_map.pbtxt --images_dir data/samples/images/ --save_output  
    
An example execution for making predictions on a video would be:

    python3 detect_objects.py --model_path models/inference_graph/saved_model --path_to_labelmap models/label_map.pbtxt --video_input --video_path data/video_samples/1.mov --save_output

To execute the script you need to specify the following parameters:
* ```--model_path``` System path to the frozen detection model, default=```models/efficientdet_d0_coco17_tpu-32/saved_model```
* ```--path_to_labelmap``` Path to the labelmap (.pbtxt) file
* ```--class_ids``` IDs of classes to detect, expects string with IDs separated by ","
* ```--threshold``` Detection Ttreshold, default=```0.4```
* ```--images_dir``` Directory to input images, default=```'data/samples/images/```
* ```--video_path``` Path to input video
* ```--output_directory``` Path to output images/video, default=```data/samples/output```
* ```--video_input``` Flag for video input, default=```False```
* ```--save_output``` Flag for saveing images and video with visualized detections, default=```False```

#### Faster R-CNN from scratch in Jupyter Notebook ####

We have also implemented the Faster R-CNN algorithm from scratch.

    Faster_RCNN_Final.ipynb

However, the training was very slow because of bad vectorization inside the source code. This is also the reason why we decided to use the Faster R-CNN API.

## Tools ## 
* Python 3
* PyTorch Framework
* OpenCV
* TensorFlow 2

## Results ##
For the evaluation and the comparison of YOLO and Faster R-CNN, we measured the FPS and the mAP on the test dataset of BDD100K using a NVDIA V100 SXM2 32GB GPU.

| Architecture| mAP % | FPS |
| -------------|:-----:| -----
| YOLO     | 18,6 | 212,4 |
| Faster R-CNN| 41,8 | 17,1 |

### YOLO ###
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/YOLO/yolo_1.gif">
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/YOLO/yolo_2.gif">
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/YOLO/yolo_3.gif">
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/YOLO/yolo_4.gif">

### Faster R-CNN ###
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_1.gif">
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_2.gif">
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_3.gif">
<img align="center" width="1000" height="" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_4.gif">

### YOLO (left) & Faster R-CNN (right) comparison ### 
<img align="left" width="390" height="" src="Result%20images%20and%20videos/YOLO/yolo_5.gif">
<img align="right" width="390" height="" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_5.gif">
<img align="left" width="390" height="" src="Result%20images%20and%20videos/YOLO/yolo_6.gif">
<img align="right" width="390" height="" src="Result%20images%20and%20videos/Faster%20R-CNN/faster_rcnn_6.gif">
