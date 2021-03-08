from model import YOLOv1
from loss import YOLO_Loss
from dataset import DataLoader
from utils import load_checkpoint, save_checkpoint
import torch.optim as optim
import torch
import time
import os
import argparse


# Argparse to start the YOLO training
ap = argparse.ArgumentParser()
ap.add_argument("-tip", "--train_img_files_path", default="bdd100k/images/100k/train/", 
                help="path to the train image folder")
ap.add_argument("-ttp", "--train_target_files_path", 
                default="bdd100k_labels_release/bdd100k/labels/det_v2_train_release.json", 
                help="path to json file containing the train labels")
ap.add_argument("-lr", "--learning_rate", default=1e-5, help="learning rate")
ap.add_argument("-bs", "--batch_size", default=10, help="batch size")
ap.add_argument("-ne", "--number_epochs", default=100, help="amount of epochs")
ap.add_argument("-ls", "--load_size", default=1000, 
                help="amount of batches which are being loaded in one take")
ap.add_argument("-nb", "--number_boxes", default=2, 
                help="amount of bounding boxes which should be predicted")
ap.add_argument("-lc", "--lambda_coord", default=5, 
                help="hyperparameter penalizeing predicted bounding boxes in the loss function")
ap.add_argument("-ln", "--lambda_noobj", default=0.5, 
                help="hyperparameter penalizeing prediction confidence scores in the loss function")
ap.add_argument("-lm", "--load_model", default=1, 
                help="1 if the model weights should be loaded else 0")
ap.add_argument("-lmf", "--load_model_file", default="YOLO_bdd100k.pt", 
                help="name of the file containing the model weights")
args = ap.parse_args()

# Dataset parameters
train_img_files_path = args.train_img_files_path
train_target_files_path = args.train_target_files_path
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                "truck", "train", "other person", "bus", "car", "rider", "motorcycle", 
                "bicycle", "trailer"]

# Hyperparameters
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
num_epochs = int(args.number_epochs)
load_size = int(args.load_size)
split_size = 14
num_boxes = int(args.number_boxes)
lambda_coord = float(args.lambda_coord)
lambda_noobj = float(args.lambda_noobj)

# Other parameters
cell_dim = int(448/split_size)
num_classes = len(category_list)
load_model = int(args.load_model)
load_model_file = args.load_model_file


def TrainNetwork(num_epochs, split_size, batch_size, load_size, num_boxes, num_classes, 
                 train_img_files_path, train_target_files_path, category_list, model, 
                 device, optimizer, load_model_file, lambda_coord, lambda_noobj):
    """
    Starts the training process of the model.
    
    Parameters:
        num_epochs (int): Amount of epochs for training the model.
        split_size (int): Size of the grid which is applied to the images.
        batch_size (int): Batch size.
        load_size (int): Amount of batches which are loaded in one function call.
        num_boxes (int): Amount of boxes which are being predicted per grid cell.
        num_classes (int): Amount of classes which are being predicted.        
        train_img_files_path (str): System path to the image folder containing 
        the train images.
        train_target_files_path (str): System path to the target folder containing 
        the json file with the ground-truth labels.
        category_list (list): A list containing all ground-truth classes.
        model (): The YOLOv1-model. 
        device (): The device used for training.
        optimizer (): Algorithm for updating the model weights.
        load_model_file (str): Name of the file used to store/load train checkpoints.
        lambda_coord (float): Hyperparameter for the loss regarding the bounding 
        box coordinates.
        lambda_noobj (float): Hyperparameter for the loss in case there is no 
        object in that cell.
    """
    
    model.train()
    
    # Initialize the DataLoader for the train dataset
    data = DataLoader(train_img_files_path, train_target_files_path, category_list, 
                      split_size, batch_size, load_size)
    
    loss_log = {} # Used for tracking the loss
    torch.save(loss_log, "loss_log.pt") # Initialize the log file
    
    for epoch in range(num_epochs):
        epoch_losses = [] # Stores the loss progress 
    
        print("DATA IS BEING LOADED FOR A NEW EPOCH")
        print("")
        data.LoadFiles() # Resets the DataLoader for a new epoch

        while len(data.img_files) > 0:
            all_batch_losses = 0. # Used to track the training loss
            
            print("LOADING NEW BATCHES")            
            print("Remaining files:" + str(len(data.img_files)))
            print("")
            data.LoadData() # Loads new batches 
            
            for batch_idx, (img_data, target_data) in enumerate(data.data):
                img_data = img_data.to(device)
                target_data = target_data.to(device)
                
                optimizer.zero_grad()
                
                predictions = model(img_data)
                
                yolo_loss = YOLO_Loss(predictions, target_data, split_size, num_boxes, 
                                      num_classes, lambda_coord, lambda_noobj)
                yolo_loss.loss()
                loss = yolo_loss.final_loss          
                all_batch_losses += loss.item()
                
                loss.backward()
                optimizer.step()

                print('Train Epoch: {} of {} [Batch: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch+1, num_epochs, batch_idx+1, len(data.data),
                    (batch_idx+1) / len(data.data) * 100., loss))
                print('')

            epoch_losses.append(all_batch_losses / len(data.data))
            print("Loss progress so far:", epoch_losses)
            print("")
  
        loss_log = torch.load('loss_log.pt')
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        loss_log['Epoch: ' + str(epoch+1)] = mean_loss
        torch.save(loss_log, 'loss_log.pt')     
        print(f"Mean loss for this epoch was {sum(epoch_losses)/len(epoch_losses)}")
        print("")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=load_model_file)

        time.sleep(10)
        
        
def main():    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  
    device = torch.device('cuda')
    
    # Initialize model
    model = YOLOv1(split_size, num_boxes, num_classes).to(device)
    
    # Define the learning method for updating the model weights
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Load model and optimizer parameters
    if load_model:
        print("##################### LOADING YOLO MODEL ######################")
        print("")
        load_checkpoint(torch.load(load_model_file), model, optimizer)

    # Start the training process
    print("###################### STARTING TRAINING ######################")
    print("")
    TrainNetwork(num_epochs, split_size, batch_size, load_size, num_boxes, 
                 num_classes, train_img_files_path, train_target_files_path, 
                 category_list, model, device, optimizer, load_model_file, 
                 lambda_coord, lambda_noobj)


if __name__ == "__main__":
    main()