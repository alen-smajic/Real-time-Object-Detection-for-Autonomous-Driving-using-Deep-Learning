import torch
import torch.optim as optim
from model import YOLOv1
from utils import load_checkpoint
from train import TrainNetwork
from validation import validate


# Dataset parameters
train_img_files_path = "bdd100k/images/100k/train/"
train_target_files_path = "bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json"
val_img_files_path = "bdd100k/images/100k/val/"
val_target_files_path = "bdd100k_labels_release/bdd100k/labels/det_v2_val_release.json"
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", "truck", "train", "other person", "bus", "car", "rider", "motorcycle", "bicycle", "trailer"]

# Hyperparameters
learning_rate = 1e-5
weight_decay = 0
batch_size = 10
num_epochs = 2
load_size = 1000
split_size = 7
num_boxes = 2
lambda_coord = 5
lambda_noobj = 0.5
iou_threshold = 0.5
threshold = 0.5

# Other parameters
cell_dim = int(448/split_size)
num_classes = len(category_list)
load_model = False
load_model_file = "YOLO_bdd100k.pth"


def main():    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    # Initialize model
    model = YOLOv1(split_size, num_boxes, num_classes).to(device)
    
    # Define the learning method for updating the model weights
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Load model and optimizer parameters
    if load_model:
        print("###################### LOADING YOLO MODEL ######################")
        print("")
        load_checkpoint(torch.load(load_model_file), model, optimizer)
    
    # Start the training process
    print("###################### STARTING TRAINING ######################")
    print("")
    TrainNetwork(    
        num_epochs, 
        split_size, 
        batch_size, 
        load_size, 
        num_boxes, 
        num_classes,    
        train_img_files_path, 
        train_target_files_path, 
        category_list,     
        model, 
        device, 
        optimizer, 
        load_model_file,
        lambda_coord, 
        lambda_noobj, 
    )
    
    # Start the validation process to calculate the final model performance on the test set
    print("###################### STARTING VALIDATION ######################")
    print("")
    validate(val_img_files_path, val_target_files_path, category_list, split_size, batch_size, load_size, model, cell_dim, num_boxes, num_classes, device, iou_threshold, threshold)

if __name__ == "__main__":
    main()