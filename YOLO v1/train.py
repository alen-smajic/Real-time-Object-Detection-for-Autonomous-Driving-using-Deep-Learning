import torch
from loss import YOLO_Loss
from dataset import DataLoader
from utils import save_checkpoint
import time


def TrainNetwork(    
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
    ):
    """
    Starts the training process of the model.
    
    Parameters:
        num_epochs (int): Amount of epochs for training the model.
        split_size (int): Size of the grid which is applied to the images.
        batch_size (int): Batch size.
        load_size (int): Amount of batches which are loaded in one function call.
        num_boxes (int): Amount of boxes which are being predicted per grid cell.
        num_classes (int): Amount of classes which are being predicted.        
        train_img_files_path (str): System path to the image folder containing the train images.
        train_target_files_path (str): System path to the target folder containing the json file with the ground-truth labels.
        category_list (list): A list containing all ground-truth classes.
        model (): The YOLOv1-model. 
        device (): The device used for training.
        optimizer (): Algorithm for updating the model weights.
        load_model_file (str): Name of the file used to store/load train checkpoints.
        lambda_cooord (float): Hyperparameter for the loss regarding the bounding box coordinates.
        lambda_noobj (float): Hyperparameter for the loss in case there is no object in that cell.
    """
    
    # Initialize the DataLoader for the train dataset
    data = DataLoader(train_img_files_path, train_target_files_path, category_list, split_size, batch_size, load_size)
    
    for epoch in range(num_epochs):
        model.train()
        
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
                
                predictions = model(img_data)
                
                yolo_loss = YOLO_Loss(predictions, target_data, split_size, num_boxes, num_classes, lambda_coord, lambda_noobj)
                yolo_loss.loss()
                loss = yolo_loss.final_loss          
                all_batch_losses += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Train Epoch: {} of {} [Batch: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch+1, num_epochs, batch_idx+1, len(data.data),
                    (batch_idx+1) / len(data.data) * 100., loss))
                print('')

            epoch_losses.append(all_batch_losses / len(data.data))
            print("Loss progress so far:", epoch_losses)
            print("")
  
        torch.save(epoch_losses, "epoch_" + epoch + "_losses.pt")     
        print(f"Mean loss for this epoch was {sum(epoch_losses)/len(epoch_losses)}")
        print("")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=load_model_file)

        time.sleep(10)