import torch
from dataset import DataLoader
from utils import MidtoCorner, IoU
from collections import Counter


def validate(val_img_files_path, val_target_files_path, category_list, split_size, batch_size, load_size, model, cell_dim, num_boxes, num_classes, device, iou_threshold, threshold):
    
    model.eval()
    
    data = DataLoader(val_img_files_path, val_target_files_path, category_list, split_size, batch_size, load_size)
    
    print("DATA IS BEING LOADED FOR VALIDATION")
    print("")
    data.LoadFiles()
    
    all_pred_boxes = []
    all_target_boxes = []
    train_idx = 0

    while len(data.img_files) > 0:
        if len(data.img_files) < 8000:
            break
        print("LOADING NEW VALIDATION BATCHES")            
        print("Remaining validation files:" + str(len(data.img_files)))
        print("")
        data.LoadData()
            
        for batch_idx, (img_data, target_data) in enumerate(data.data):
            img_data = img_data.to(device)
            target_data = target_data.to(device)
    
            with torch.no_grad():
                predictions = model(img_data)
                
            pred_boxes = extract_boxes(predictions, num_classes, num_boxes, cell_dim)
            target_boxes = extract_boxes(target_data, num_classes, num_boxes, cell_dim)
            
            for sample_idx in range(batch_size):
                nms_boxes = non_max_suppression(pred_boxes[sample_idx], iou_threshold, threshold)                  
                
                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)
                
                for box in target_boxes[sample_idx]:
                    if box[1] > threshold:
                        all_target_boxes.append([train_idx] + box)
            
                train_idx += 1
                
    mean_avg_prec = mean_average_precision(all_pred_boxes, all_target_boxes, iou_threshold, box_format="corner")
    print(f"Train mAP: {mean_avg_prec}")
            
    
    
                
                                        
def extract_boxes(yolo_tensor, num_classes, num_boxes, cell_dim):
    all_bboxes = []
            
    for sample_idx in range(yolo_tensor.shape[0]):
        bboxes = []
        for cell_h in range(yolo_tensor.shape[1]):
            for cell_w in range(yolo_tensor.shape[2]):
                best_class = 0
                max_conf = 0
                for class_idx in range(num_classes):
                    if yolo_tensor[sample_idx, cell_h, cell_w, num_boxes*5+class_idx] > max_conf:
                        max_conf = yolo_tensor[sample_idx, cell_h, cell_w, num_boxes*5+class_idx]
                        best_class = class_idx
                for box_idx in range(num_boxes):
                    cords = MidtoCorner(yolo_tensor[sample_idx, cell_h, cell_w, box_idx*5+1:box_idx*5+5], cell_h, cell_w, cell_dim)
                    x1 = cords[0]
                    y1 = cords[1]
                    x2 = cords[2]
                    y2 = cords[3]
                    bboxes.append([best_class, yolo_tensor[sample_idx, cell_h, cell_w, box_idx*5], x1, y1, x2, y2])
                all_bboxes.append(bboxes)
    return all_bboxes


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IoU(
                chosen_box[2:],
                box[2:]
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(
                    gt[3:],detection[3:]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)