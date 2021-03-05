from torchvision import transforms
from model import YOLOv1
from PIL import Image
import argparse
import time
import os
import cv2
import torch


# All BDD100K (dataset) classes and the corresponding class colors for drawing 
# the bounding boxes 
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                 "truck", "train", "other person", "bus", "car", "rider", 
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                  (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                  (127,0,255),(0,128,255),(128,128,128)]

# Argparse to apply YOLO algorithm to a video file from the console
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to the modell weights")
ap.add_argument("-t", "--threshold", default=0.5, 
                help="threshold for the confidence score of the bouding box prediction")
ap.add_argument("-ss", "--split_size", default=14, 
                help="split size of the grid which is applied to the image")
ap.add_argument("-nb", "--num_boxes", default=2, 
                help="number of bounding boxes which are being predicted")
ap.add_argument("-nc", "--num_classes", default=13, 
                help="number of classes which are being predicted")
ap.add_argument("-i", "--input", required=True, help="path to your input video")
ap.add_argument("-o", "--output", required=True, help="path to your output video")
args = ap.parse_args()


def main(): 
    print("")
    print("##### YOLO OBJECT DETECTION FOR VIDEOS #####")
    print("")    
    print("Loading the model")
    print("...")
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  
    device = torch.device('cuda')
    model = YOLOv1(int(args.split_size), int(args.num_boxes), int(args.num_classes)).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Amount of YOLO parameters: " + str(num_param))
    print("...")
    print("Loading model weights")
    print("...")
    weights = torch.load(args.weights)
    model.load_state_dict(weights["state_dict"])
    model.eval()
    
    # Transform is applied to the input frames
    # It resizes the image and converts it into a tensor
    transform = transforms.Compose([
        transforms.Resize((448,448), Image.NEAREST),
        transforms.ToTensor(),
        ])  
    
    print("Loading input video file")
    print("...")
    vs = cv2.VideoCapture(args.input)
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Defining the output video file
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30, 
            (frame_width, frame_height))
    
    # Used to scale the bounding box predictions to the original input frame
    # (448 is the dimension of the input image for the model)
    ratio_x = frame_width/448
    ratio_y = frame_height/448
    
    idx = 1 # Used to track how many frames have been already processed
    sum_fps = 0 # Used to track the average FPS at the end
    amount_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) # Amount of frames
    
    while True:              
        grabbed, frame = vs.read()
        if not grabbed:
            break 
        
        # Logging the amount of processed frames 
        print("Loading frame " + str(idx) + " out of " + str(amount_frames))  
        print("Percentage done: {0:.0%}".format(idx/amount_frames)) 
        print("") 
        
        idx += 1 # Frame index
        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0).to(device)
        img = cv2.UMat(frame)

        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor) # Makes a prediction on the input frame
            curr_fps = int(1.0 / (time.time() - start_time)) # Prediction FPS
            sum_fps += curr_fps
            print("FPS for YOLO prediction: " + str(curr_fps))
            print("")

        # Extracts the class index with the highest confidence scores
        corr_class = torch.argmax(output[0,:,:,10:23], dim=2)

        for cell_h in range(output.shape[1]):
            for cell_w in range(output.shape[2]):                
                # Determines the best bounding box prediction 
                best_box = 0
                max_conf = 0
                for box in range(int(args.num_boxes)):
                    if output[0, cell_h, cell_w, box*5] > max_conf:
                        best_box = box
                        max_conf = output[0, cell_h, cell_w, box*5]
                
                # Checks if the confidence score is above the specified threshold
                if output[0, cell_h, cell_w, best_box*5] >= float(args.threshold):
                    # Extracts the box confidence score, the box coordinates and class
                    confidence_score = output[0, cell_h, cell_w, best_box*5]
                    center_box = output[0, cell_h, cell_w, best_box*5+1:best_box*5+5]
                    best_class = corr_class[cell_h, cell_w]
                    
                    # Transforms the box coordinates into pixel coordinates
                    centre_x = center_box[0]*32 + 32*cell_w
                    centre_y = center_box[1]*32 + 32*cell_h
                    width = center_box[2] * 448
                    height = center_box[3] * 448
                    
                    # Calculates the corner values of the bounding box
                    x1 = int((centre_x - width/2) * ratio_x)
                    y1 = int((centre_y - height/2) * ratio_y)
                    x2 = int((centre_x + width/2) * ratio_x)
                    y2 = int((centre_y + height/2) * ratio_y)

                    # Draws the bounding box with the corresponding class color
                    # around the object
                    cv2.rectangle(img, (x1,y1), (x2,y2), category_color[best_class], 1)
                    # Generates the background for the text, painted in the corresponding
                    # class color and the text with the class label including the 
                    # confidence score
                    labelsize = cv2.getTextSize(category_list[best_class], 
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1-20), (x1+labelsize[0][0]+45,y1), 
                                  category_color[best_class], -1)
                    cv2.putText(img, category_list[best_class] + " " + 
                                str(int(confidence_score.item()*100)) + "%", (x1,y1-5), 
                                cv2.FONT_HERSHEY_DUPLEX , 0.5, (0,0,0), 1, cv2.LINE_AA)
                    # Generates a small window in the top left corner which 
                    # displays the current FPS for the prediction
                    cv2.putText(img, str(curr_fps) + "FPS", (25, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        out.write(img) # Stores the frame with the predictions on a new mp4 file
    print("Average FPS was: " + str(int(sum_fps / amount_frames))) 
    print("")

if __name__ == '__main__':
    main()