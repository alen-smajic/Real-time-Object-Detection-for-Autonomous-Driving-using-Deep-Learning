import torch
from utils import IoU, MidtoCorner


class YOLO_Loss():
    """
    Used to calculate the loss for the YOLO-model using a batch of predictions 
    and the corresponding ground-truth labels. 
    """
    
    def __init__(self, predictions, targets, split_size, num_boxes, num_classes, 
                 lambda_coord, lambda_noobj):
        """
        Initialize the parameters for calculating the loss value.
        
        Parameters:
            predictions (tensor): A tensor containing a mini-batch of predicted samples.
            targets (tensor): A tensor containing a mini-batch of ground-truth labels.
            split_size (int): Specifies the size of the grid which is applied to the image.
            num_boxes (int): Amount of bounding boxes which are predicted by the YOLO-model.
            num_classes (int): Amount of classes which are being predicted.
            lambda_cooord (float): Hyperparameter for the loss regarding the bounding
            box coordinates.
            lambda_noobj (float): Hyperparameter for the loss in case there is no object 
            in the cell.
        """
        
        self.predictions = predictions
        self.targets = targets
        self.split_size = split_size
        self.cell_dim = int(448 / split_size) # Dimension of a single cell
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.final_loss = 0 # Here will the final value of the loss function be stored
        
        
    def loss(self):
        """
        Main function for calculating the loss. Stores the calculated loss inside 
        the final_loss atribute.
        """
        
        for sample in range(self.predictions.shape[0]):
            mid_loss = 0 # Loss of the centre coordinates
            dim_loss = 0 # Loss of the width and height values
            conf_loss = 0 # Loss of the confidence score when there is an object in the cell
            conf_loss_noobj = 0 # Loss of the confidence score when there is no object in the cell
            class_loss = 0 # Loss of the class score
            for cell_h in range(self.split_size):
                for cell_w in range(self.split_size):
                    # Check if the current cell contains an object
                    if self.targets[sample, cell_h, cell_w, 0] != 1:
                        conf_loss_noobj += self.noobj_loss(sample, cell_h, cell_w)
                    else:
                        mid_loss_local, dim_loss_local, conf_loss_local, class_loss_local = self.obj_loss(sample, cell_h, cell_w)
                        mid_loss += mid_loss_local
                        dim_loss += dim_loss_local
                        conf_loss += conf_loss_local
                        class_loss += class_loss_local
                        
            # Calculate the final loss by summing the other losses and applying 
            # the hyperparameters lambda_coord and lambda_noobj
            self.final_loss += self.lambda_coord*mid_loss + self.lambda_coord*dim_loss 
            + self.lambda_noobj*conf_loss_noobj + conf_loss + class_loss
                        
                    
    def noobj_loss(self, sample, cell_h, cell_w):
        """
        Calculates the loss value for a single cell in case there is no 
        ground-truth object in that cell.
        
        Parameters:
            sample (int): The index of the current sample from the batch.
            cell_h (int): Index of the cell coordinate.
            cell_w (int): Index of the cell coordinate.
            
        Return:
            loss_value (float): The value of the loss with respect to the cell.
        """
        
        loss_value = 0.
        for box in range(self.num_boxes):
            loss_value += (0 - self.predictions[sample, cell_h, cell_w, box*5])**2
        return loss_value

        
    def obj_loss(self, sample, cell_h, cell_w): 
        """
        Calculates the loss value for a single cell in case there is a ground-truth 
        object in that cell.
        
        Parameters:
            sample (int): The index of the current sample from the batch.
            cell_h (int): Index of the cell coordinate.
            cell_w (int): Index of the cell coordinate.
            
        Return:
            mid_loss_local (float): Loss value for the mid coordinates of the 
            bounding box.
            dim_loss_local (float): Loss value for the height and width coordinates 
            of the bounding box.
            conf_loss_local (float): Loss value for the confidence score.
            class_loss_local (float): Loss value for the class scores.
        """
        
        # Finds the box with the highest IoU with respect to the ground-truth and 
        # stores its index in best_box
        if self.num_boxes != 1:
            best_box = self.find_best_box(sample, cell_h, cell_w)
        else:
            best_box = 0
        
        # Calculates the loss for the centre coordinates
        x_loss = torch.square(self.targets[sample, cell_h, cell_w, 1] - 
                              self.predictions[sample, cell_h, cell_w, 1+best_box*5])        
        y_loss = torch.square(self.targets[sample, cell_h, cell_w, 2] - 
                              self.predictions[sample, cell_h, cell_w, 2+best_box*5])
        mid_loss_local = x_loss + y_loss
                
        # Calculates the loss for the width and height values
        w_loss = torch.square(torch.sqrt(self.targets[sample, cell_h, cell_w, 3]) - 
                              torch.sqrt(self.predictions[sample, cell_h, cell_w, 3+best_box*5]))
        h_loss = torch.square(torch.sqrt(self.targets[sample, cell_h, cell_w, 4]) - 
                              torch.sqrt(self.predictions[sample, cell_h, cell_w, 4+best_box*5]))
        dim_loss_local = w_loss + h_loss
                
        # Calculates the loss of the confidence score
        conf_loss_local = torch.square(1 - self.predictions[sample, cell_h, cell_w, best_box*5])
                
        # Calculates the loss for the class scores
        class_loss_local = 0.
        for c in range(self.num_classes):
            class_loss_local += torch.square(self.targets[sample, cell_h, cell_w, 5+c] - 
                                             self.predictions[sample, cell_h, cell_w, 5*self.num_boxes+c])
            
        return mid_loss_local, dim_loss_local, conf_loss_local, class_loss_local
                        
    
    def find_best_box(self, sample, cell_h, cell_w):
        """
        Finds the bounding box with the highest IoU with respect to the 
        ground-truth bounding box.
        
        Parameters:
            sample (int): The index of the current sample from the batch.
            cell_h (int): Index of the cell coordinate.
            cell_w (int): Index of the cell coordinate.
            
        Returns:
            best_box (int): The index of the bounding box with the highest IoU.
        """
        
        # Transform the box coordinates into the corner format
        t_box_coords = MidtoCorner(self.targets[sample, cell_h, cell_w, 1:5], 
                                   cell_h, cell_w, self.cell_dim)
        
        best_box = 0
        max_iou = 0.
        for box in range(self.num_boxes):
            # Transform the box coordinates into the corner format
            p_box_coords = MidtoCorner(self.predictions[sample, cell_h, cell_w, 
                                                        1+box*5:5+box*5], cell_h, cell_w, self.cell_dim)
                    
            box_score = IoU(t_box_coords, p_box_coords)
            if box_score > max_iou:
                max_iou = box_score
                best_box = box # Store the box index with the highest IoU          
                
        return best_box