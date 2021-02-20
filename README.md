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

#### Detect Bounding Boxes ####

We provide 2 scripts to detect bounding boxes on videos and images:
* ```YOLO_to_video.py```
* ```YOLO_to_image.py```

To start the script you need to specify the following parameters:
* ```--weights``` path to the modell weights 
* ```--threshold``` threshold for the confidence score of the bounding box prediction
* ```--input``` path to your input image/video 
* ```--output``` path to your output image/video (in case of video include .mp4 at the end)

An example execution for images would be:

    python3 YOLO_to_image.py -w /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/YOLO_bdd100k.pt -t 0.5 -i /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/test.jpg -o /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/output.png

An example execution for video would be:

    python3 YOLO_to_video.py -w /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/YOLO_bdd100k.pt -t 0.5 -i /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/test.mov -o /home/alen_smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning/YOLO/Inference_YOLO/output.mp4

## Publications ##
  
## Tools ## 
* Python 3
* PyTorch Framework
* OpenCV

## Results ##
<img align="center" width="1000" height="" src="Preliminary%20results/current_results.gif">
<img align="center" width="1000" height="" src="Preliminary%20results/Screenshot%202021-01-12%20131231.png">

<img align="left" width="390" height="" src="Preliminary%20results/Screenshot%202021-01-12%20131156.png">
<img align="right" width="390" height="" src="Preliminary%20results/Screenshot%202021-01-12%20131214.png">
<img align="left" width="390" height="" src="Preliminary%20results/Screenshot%202021-01-12%20131303.png">
<img align="right" width="390" height="" src="Preliminary%20results/Screenshot%202021-01-12%20131317.png">
