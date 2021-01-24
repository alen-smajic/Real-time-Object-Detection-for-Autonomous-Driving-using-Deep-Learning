from loss import YOLO_Loss
from dataset import DataLoader
from utils import save_checkpoint


def TrainNetwork(num_epochs, train_img_files_path, train_target_files_path, category_list, split_size, batch_size, load_size, model, device, num_boxes, num_classes, lambda_coord, lambda_noobj, optimizer, load_model_file):
    """
    Starts the training process of the model.
    
    Parameters:
        num_epochs (int): Amount of epochs for training the model.
    """
    
    data = DataLoader(train_img_files_path, train_target_files_path, category_list, split_size, batch_size, load_size)
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_mean_loss = []
    
        print("DATA IS BEING LOADED FOR A NEW EPOCH")
        print("")
        data.LoadFiles()

        while len(data.img_files) > 0:
            print("LOADING NEW BATCHES")            
            print("Remaining files:" + str(len(data.img_files)))
            print("")
            data.LoadData()
            
            for batch_idx, (img_data, target_data) in enumerate(data.data):
                img_data = img_data.to(device)
                target_data = target_data.to(device)    
                predictions = model(img_data)                
                yolo_loss = YOLO_Loss(predictions, target_data, split_size, num_boxes, num_classes, lambda_coord, lambda_noobj)
                yolo_loss.loss()
                loss = yolo_loss.final_loss                
                epoch_mean_loss.append(loss.item())                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Train Epoch: {} of {} [Batch: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch+1, num_epochs, batch_idx+1, len(data.data),
                    (batch_idx+1) / len(data.data) * 100., loss))
                print('')
                
        print(f"Mean loss was {sum(epoch_mean_loss)/len(epoch_mean_loss)}")
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=load_model_file)
        import time
        time.sleep(10)