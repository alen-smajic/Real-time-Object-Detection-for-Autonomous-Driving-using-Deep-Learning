import torch


def IoU(target, prediction):
    """
    Calculates the Intersection over Union of two bounding boxes.
        
    Parameters:
    target (list): A list with bounding box coordinates in the corner format.
    predictions (list): A list with bounding box coordinates in the corner format.
    
    Returns:
    iou_value (float): The score of the IoU over the two boxes.
    """
        
    # Calculate the corner coordinates of the intersection
    i_x1 = max(target[0], prediction[0])
    i_y1 = max(target[1], prediction[1])
    i_x2 = min(target[2], prediction[2])
    i_y2 = min(target[3], prediction[3])

    intersection = max(0,(i_x2-i_x1)) * max(0,(i_y2-i_y1))    
    union = ((target[2]-target[0]) * (target[3]-target[1])) + ((prediction[2]-prediction[0]) * 
                                                               (prediction[3]-prediction[1])) - intersection

    iou_value = intersection / union    
    return iou_value


def MidtoCorner(mid_box, cell_h, cell_w, cell_dim):
    """
    Transforms bounding box coordinates which are in the mid YOLO format into the 
    common corner format with the correct pixel locations.
    
    Parameters:
        mid_box (list): Bounding box coordinates which are in the mid YOLO format 
        [x_mid, y_mid, width, height].
        cell_h (int): Height index of the cell with the bounding box.
        cell_w (int): Width index of the cell with the bounding box.
        cell_dim (int): Dimension of a single cell.
        
    Returns:
        corner_box (list): A list containing the coordinates of the bounding 
        box in the common corner format [x1, y2, x2, y2].
    """
    
    # Transform the coordinates from the YOLO format into normal pixel values
    centre_x = mid_box[0]*cell_dim + cell_dim*cell_w
    centre_y = mid_box[1]*cell_dim + cell_dim*cell_h
    width = mid_box[2] * 448
    height = mid_box[3] * 448
    
    # Calculate the corner values of the bounding box
    x1 = int(centre_x - width/2)
    y1 = int(centre_y - height/2)
    x2 = int(centre_x + width/2)
    y2 = int(centre_y + height/2)
    
    corner_box = [x1,y1,x2,y2]  
    return corner_box    


def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads the model weights and optimizer state (the checkpoint).
    
    Parameters:
        checkpoint (string): The file from which the checkpoint is being loaded.
        model (): The model which is being overwritten by the checkpoint.
        optimizer (): The optimizer which is being overwritten by the checkpoint.
    """
    print("=> Loading checkpoint")
    print("")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    

def save_checkpoint(state, filename):
    """
    Saves the model weights and optimizer state (the checkpoint).
    
    Parameters:
        state (dict): A dictionary containing the model- and optimizer-state.
        filename (string): The file to which the checkpoint is saved.
    """
    print("=> Saving checkpoint")
    print("")
    torch.save(state, filename)