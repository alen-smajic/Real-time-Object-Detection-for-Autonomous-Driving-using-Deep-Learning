# Real-time Object Detection for Autonomous Driving using Deep Learning, Goethe University Frankfurt (Fall 2020)

## General Information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Instructors:**
* [Prof. Dr. Gemma Roig](http://www.cvai.cs.uni-frankfurt.de/team.html), email: roig@cs.uni-frankfurt.de
* Dr. Iuliia Pliushch
* Kshitij Dwivedi
* Matthias Fulde

**Institutions:**
  * **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
  * **[Computational Vision & Artificial Intelligence](http://www.cvai.cs.uni-frankfurt.de/index.html)**

**Project team (A-Z):**
* Duy Anh Tran
* Pascal Fischer
* Alen Smajic
* Yujin So

## Project Description ##
Datasets drive vision progress, yet existing driving datasets are limited in terms of visual content, scene variation, the richness of annotations, and the geographic distribution and supported tasks to study multitask learning for autonomous driving.
In 2018 Yu et al. released BDD100K, the largest driving video dataset with 100K videos and 10 tasks to evaluate the progress of image recognition algorithms on autonomous driving. The dataset possesses geographic, environmental, and weather diversity, which is useful for training models that are less likely to be surprised by new conditions. Provided are bounding box annotations of 13 categories for each of the reference frames of 100K videos and 2D bounding boxes annotated on 100.000 images for "other vehicle", "pedestrian", "traffic light", "traffic sign", "truck", "train", "other person", "bus", "car", "rider", "motorcycle", "bicycle", "trailer"


This project is still in development.

- [x] Find a dataset
- [x] Implement the Data loader
- [x] Implement the YOLO model
- [x] Implement the YOLO Loss function 
- [x] Implement the training loop
- [x] Evaluate the performance

### Dataset ###

You can download the dataset [here](https://bdd-data.berkeley.edu/)

### YOLO ###

#### Training YOLO #####

To train the model use the script:
* ```train.py```

To execute the script you need to specify the following parameters:
* ```--train_img_files_path``` ```-tip``` (default: bdd100k/images/100k/train/) path to the train image folder
* ```--train_target_files_path``` ```-ttp``` (default: bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json) path to json file containing the train labels
* ```--category_list``` ```-cl``` (default: ["other vehicle", "pedestrian", "traffic light", "traffic sign", "truck", "train", "other person", "bus", "car", "rider", "motorcycle", "bicycle", "trailer"]) list containing all string names of the classes
* ```--learning_rate``` ```-lr``` (default: 1e-5) learning rate
* ```--batch_size``` ```-bs``` (default: 10) batch size
* ```--number_epochs``` ```-ne``` (default: 100) amount of epochs
* ```--load_size``` ```-ls``` (default: 1000) amount of batches which are being loaded in one take
* ```--number_boxes``` ```-nb``` (default: 2) amount of bounding boxes which should be predicted
* ```--lambda_coord``` ```-lc``` (default: 5) hyperparameter penalizeing predicted bounding boxes in the loss function
* ```--lambda_noobj``` ```-ln``` (default: 0.5) hyperparameter penalizeing prediction confidence scores in the loss function
* ```--load_model``` ```-lm``` (default: 1) 1 if the model weights should be loaded else 0
* ```--load_model_file``` ```-lmf``` (default: "YOLO_bdd100k.pt") name of the file containing the model weights

An example execution for training would be:

    python3 train.py -tip bdd100k/images/100k/train/ -ttp bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json -lr 1e-5 -bs 10 -ne 100 -ls 1000 -nb 2 -lc 5 -ln 0.5 -lm 1 -lmf YOLO_bdd100k.pt 

#### Detecting Bounding Boxes with YOLO ####

We provide 2 scripts to detect bounding boxes on images and videos:
* ```YOLO_to_video.py```
* ```YOLO_to_image.py```

To execute the script you need to specify the following parameters:
* ```--weights``` ```-w``` path to the modell weights 
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

## Publications ##
  
## Tools ## 
* Python 3
* PyTorch Framework
* OpenCV

## Results ##
### YOLO ###
<img align="center" width="1000" height="" src="Result%20images/YOLO/output1.gif">
<img align="center" width="1000" height="" src="Result%20images/YOLO/output2.gif">
<img align="center" width="1000" height="" src="Result%20images/YOLO/output3.gif">
<img align="center" width="1000" height="" src="Result%20images/YOLO/output4.gif">
