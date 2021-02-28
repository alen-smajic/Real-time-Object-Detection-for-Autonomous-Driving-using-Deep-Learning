import torch
from torchvision import transforms
from os import listdir
from PIL import Image
import json
import random


class DataLoader():
    """
    This class uses its attributes to load the img data and transforms it into tensors.
    The tensors are then stored in mini-batches inside the data list which is the final 
    product of this class. Multiple function calls of LoadData() will initialize the data 
    list with new tensors from the image folder, excluding all the previous ones. 
    """

    def __init__(self, img_files_path, target_files_path, category_list, split_size, 
                 batch_size, load_size):
        """
        Initialize all parameters for loading and transforming the data into tensors.
        
        Parameters:
            img_files_path (string): The path to the image folder.
            target_files_path (string): The path to the json file containg the image labels.
            category_list (list): Reference list to all the class labels.
            split_size (int): Amount of grid cells.
            batch_size (int): Batch size.
            load_size (int): Amount of batches which are loaded in one function call.
        """
        
        self.img_files_path = img_files_path
        self.target_files_path = target_files_path       
        self.category_list = category_list        
        self.num_classes = len(category_list)       
        self.split_size = split_size        
        self.batch_size = batch_size      
        self.load_size = load_size
        
        self.img_files = [] # Will contain the remaining image names from the folder
        self.target_files = [] # Will contain the json elements with the ground-truth labels
        
        self.data = [] # Will contain tuples where each tuple contains a mini-batch of img and target tensors  
        self.img_tensors = [] # Used to temporary store image tensors from a single batch
        self.target_tensors = [] # Used to temporary store target tensors from a single batch
        
        # Define transform which is applied to every single image to resize 
        # and convert it into a tensor
        self.transform = transforms.Compose([
            transforms.Resize((448,448), Image.NEAREST),
            transforms.ToTensor(),
            ])
    

    def LoadFiles(self):
        """
        First function to be executed.
        Loads the images and the label file using the respective system path.
        """
            
        # All image names from the directory are loaded into the list img_files.
        self.img_files = listdir(self.img_files_path)
        
        # The json file containing the labels is loaded into the list target_files.
        f = open(self.target_files_path)
        self.target_files = json.load(f)
        
        
    def LoadData(self):
        """
        Transforms the image files and labels into tensors and loads them into batches. 
        Once a batch is full, it is stored in the data list. Fills the data list with 
        batches until the desired load_size is reached. Every image that is loaded, 
        is being excluded from future calls of this function.
        """
        
        # Reset the cache
        self.data = []    
        self.img_tensors = [] 
        self.target_tensors = [] 

        for i in range(len(self.img_files)):
            # Check if batch is full and perhaps start a new one
            if len(self.img_tensors) == self.batch_size:
                self.data.append((torch.stack(self.img_tensors), 
                                  torch.stack(self.target_tensors)))
                self.img_tensors = []
                self.target_tensors = []
                print('Loaded batch ', len(self.data), 'of ', self.load_size)
                print('Percentage Done: ', round(len(self.data)/self.load_size*100., 2), '%')
                print('')
            
            if len(self.data) == self.load_size: 
                break # The data list is full with the desired amount of batches
                
            # Extracts a single random image and the corresponding label, and 
            # transforms them into tensors. Both are appended to the img_tensors 
            # and target_tensors lists
            self.extract_image_and_label() 


    def extract_image_and_label(self):
        """
        Chooses a random image which is then being transformed into a tensor and 
        stored. Finds the corresponding label inside the json file which is then 
        being transformed into a tensor and stored. Stores both tensors inside 
        the img_tensors and target_tensors lists.
        """
        
        img_tensor, chosen_image = self.extract_image()
        target_tensor = self.extract_json_label(chosen_image)
        
        if target_tensor is not None: # Checks if the label contains any data
            self.img_tensors.append(img_tensor)
            self.target_tensors.append(target_tensor)
        else:
            print("No label found for " + chosen_image) # Log the image without label
            print("")

        
    def extract_image(self):   
        """
        Finds a random image from the train_files list and applies the transform to it. 
 
        Returns:
            img_tensor (tensor): The tensor which contains the image values.
            f (string): The string name of the image file.
        """    
        
        f = random.choice(self.img_files)
        self.img_files.remove(f)
        
        global img
        img = Image.open(self.img_files_path + f)
        img_tensor = self.transform(img) # Apply the transform to the image.
        return img_tensor, f


    def extract_json_label(self, chosen_image):
        """
        Uses the name of the image to find the corresponding json element. Then it 
        extracts the data and transforms it into a tensor which is stored inside 
        the target_tensors list.

        Parameters:
            chosen_image (string): The name of the image for which the label is needed.

        Returns:
            target_tensor (tensor): The tensor which contains the image labels.
        """
        
        for json_el in self.target_files:
            if json_el['name'] == chosen_image:
                img_label = json_el
                if img_label["labels"] is None: # Checks if a label exists for the given image
                    break
                target_tensor = self.transform_label_to_tensor(img_label)
                return target_tensor

        print("No label found for " + chosen_image) # Log the image without label
        print("")


    def transform_label_to_tensor(self, img_label):
        """
        Extracts the useful information from the json element and transforms them 
        into a tensor.
        
        Parameters:
            img_label (): A specific json element.
            
        Returns:
            target_tensor (tensor): A tensor of size (split_size,split_size,5+num_classes) 
            which is used as the target of the image.
        """
        
        # Here is the information stored
        target_tensor = torch.zeros(self.split_size, self.split_size, 5+self.num_classes)

        for labels in range(len(img_label["labels"])):
            # Store the category index if its contained within the category_list.
            category = img_label["labels"][labels]["category"]         
            if category not in self.category_list:
                continue
            ctg_idx = self.category_list.index(category)

            # Store the bounding box information and rescale it by the resize factor.
            x1 = img_label["labels"][labels]["box2d"]["x1"] * (448/img.size[0])
            y1 = img_label["labels"][labels]["box2d"]["y1"] * (448/img.size[1])
            x2 = img_label["labels"][labels]["box2d"]["x2"] * (448/img.size[0])
            y2 = img_label["labels"][labels]["box2d"]["y2"] * (448/img.size[1])

            # Transforms the corner bounding box information into a mid bounding 
            # box information
            x_mid = abs(x2 - x1) / 2 + x1
            y_mid = abs(y2 - y1) / 2 + y1
            width = abs(x2 - x1) 
            height = abs(y2 - y1) 

            # Size of a single cell
            cell_dim = int(448 / self.split_size)

            # Determines the cell position of the bounding box
            cell_pos_x = int(x_mid // cell_dim)
            cell_pos_y = int(y_mid // cell_dim)

            # Check if the cell already contains an object
            if target_tensor[cell_pos_y][cell_pos_x][0] == 1:
                continue
            
            # Stores the information inside the target_tensor
            target_tensor[cell_pos_y][cell_pos_x][0] = 1
            target_tensor[cell_pos_y][cell_pos_x][1] = (x_mid % cell_dim) / cell_dim
            target_tensor[cell_pos_y][cell_pos_x][2] = (y_mid % cell_dim) / cell_dim
            target_tensor[cell_pos_y][cell_pos_x][3] = width / 448
            target_tensor[cell_pos_y][cell_pos_x][4] = height / 448
            target_tensor[cell_pos_y][cell_pos_x][ctg_idx+5] = 1

        return target_tensor