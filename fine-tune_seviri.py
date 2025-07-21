import os
from tabnanny import check
import yaml
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore") # TODO: remove this in final release
from utils.utils import fix_lists, fix_band_list, get_data_files_fine_tune, remove_shared_months, string_to_boolean, visualize_results, create_fine_tune_plots, get_configuration, get_configuration_Hugging_Face
from utils.utils import modify_state_dict_input_channels
from utils.loss import select_loss_function
from utils.seviri_dataloader import Seviri_Dataset
from utils.metrics import metrics
from models.ViT.VisionTransformerModel import VisionTransformerConv
from models.ViT.utils import interpolate_pos_embed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm


def fine_tune(
        dataset_path_images, 
        dataset_path_masks,
        data_extension, 
        fine_tune_years, 
        fine_tune_months, 
        fine_tune_exclude_days,
        validation_years,
        validation_months,
        validation_exclude_days,
        include_bands,
        checkpoint_path,
        pre_trained_model_name,
        pre_trained_model_path,
        learning_rate,
        num_of_epochs,
        batch_size,
        patch_size,
        loss_function,
        class_weights):


        # load the data file names (paths) for pre-train data
        data_files_images_fine_tune = get_data_files_fine_tune(dataset_path_images, dataset_path_masks, data_extension, fine_tune_years, fine_tune_months, fine_tune_exclude_days)

        # load the data file names (paths) for validation data
        data_files_images_validation = get_data_files_fine_tune(dataset_path_images, dataset_path_masks, data_extension, validation_years, validation_months, validation_exclude_days)
     
        # this is a helper function, if a month from a specific year from pre-training is also present in validation data, remove it from pre-training
        data_files_images_fine_tune = remove_shared_months(data_files_images_fine_tune, data_files_images_validation)
       
        # define pre-train dataset | set "mode" to "pre-train" for pre-training
        fine_tune_dataset = Seviri_Dataset(filenames=data_files_images_fine_tune, mode='fine-tune', bands=include_bands)
        print(f"Total images for fine-tune: {len(fine_tune_dataset)}") # TODO: enable print
        
        # define validation dataset | set "mode" to "pre-training" for pre-training
        validation_dataset = Seviri_Dataset(filenames=data_files_images_validation, mode='fine-tune', bands=include_bands)
        print(f"Total images for validation: {len(validation_dataset)}")
        
        # TODO: fix: UserWarning: Creating a tensor from a list of numpy.ndarrays 
        # is extremely slow. Please consider converting the list to a single 
        # numpy.ndarray with numpy.array() before converting to a tensor. 

        # define dataloader and load files into ram
        fine_tune_data = DataLoader(fine_tune_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4)
        
        # define validation data and load into ram
        validation_data = DataLoader(validation_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4)                            
        
        
        # Seviri input data have shape of (batch, channels, height, width) = (64, 11, 32, 32)
        input_image_size      = fine_tune_dataset[0][0].shape[1] # get input size by height of the image
        input_channels_number = fine_tune_dataset[0][0].shape[0] # get input channels if the image

        # select a pre-trained model from Hugging Face or a local pre-trained model checkpoint (see fine-tune dataset.yaml)
        # if you select a model from Hugging Face then the local pre-trained model (if any) will not be loaded
        if pre_trained_model_name != "none":
            checkpoint_model_config = get_configuration_Hugging_Face(pre_trained_model_name)
        else:
            # load model from checkpoint
            checkpoint_model = torch.load(pre_trained_model_path, weights_only=False, map_location='cpu')

            # extract configuration of checkpoint model, like embed_dim size, depth, mlp ratio, etc...
            checkpoint_model_config = get_configuration(checkpoint_model)

            # get state dict from checkpoint model
            checkpoint_model_state_dict = checkpoint_model.state_dict()

        # if user input patch size if different then use users but will have to delete the proj layer!
        if checkpoint_model_config['patch_size'] != patch_size:
            patch_size = patch_size
        else:
            patch_size = checkpoint_model_config['patch_size']
        

        # check compatibility with input channels number from data and pre-trained model channels
        if checkpoint_model_config['input_channels'] != input_channels_number:
            print(f"Input Channels number is: {input_channels_number} and pre-trained model channels number is: {checkpoint_model_config['input_channels']}")

        # initiate a ViT model with ViT Encoder and Convolutional Decoder
        model = VisionTransformerConv(patch_size=patch_size,
                                      img_size=input_image_size,
                                      in_chans=input_channels_number,
                                      depth=checkpoint_model_config['depth'],
                                      embed_dim=checkpoint_model_config['embed_dim'],
                                      mlp_ratio=checkpoint_model_config['mlp_ratio'],
                                      num_heads=checkpoint_model_config['num_heads'],
                                      num_classes=2,
                                      qkv_bias=True,
                                      global_pool=False)

        # if needed increase proj layer to accept more or less input channels than pre-trained model (new layers will are initiate with zero weights)
        checkpoint_model_state_dict = modify_state_dict_input_channels(checkpoint_model_state_dict, input_channels_number, checkpoint_model_config['input_channels'])

        # interpolate pos embedings if needed
        checkpoint_model_state_dict = interpolate_pos_embed(model, checkpoint_model_state_dict) # if another ViT model remove "state_dict" or keep it if the model is saved with the backbone and not only the weights

        # if user input patch size if different then use users but will have to delete the proj layer!
        if checkpoint_model_config['patch_size'] != patch_size:
            checkpoint_model_state_dict.pop('patch_embed.proj.weight', None)
            checkpoint_model_state_dict.pop('patch_embed.proj.bias', None)

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model_state_dict, strict=False)
        #print(missing_keys, unexpected_keys) # TODO: remove this print and above but leave load_sate
        
        
        # check that gpu device is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pass the model to GPU
        model.to(device)

        # using AdamW optimizer for ViT models
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, betas=(0.95, 0.99), weight_decay=0.01)
        
        # criterion
        criterion = select_loss_function(loss_function, class_weights)

        # calculate number of steps per epoch
        steps_per_epoch = len(fine_tune_dataset)  # Assuming train_loader is your DataLoader
        total_steps = num_of_epochs * steps_per_epoch

        # using learing rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
        #scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # track metrics for fine-tune | training
        fine_tune_metrics = metrics(device=device, num_classes=2, average='none')

        # track metrics for fine-tune | validation
        validation_metrics = metrics(device=device, num_classes=2, average='none')
    

        # main fine-tune loop
        for epoch in range(num_of_epochs): # image shape = (batch, channels, height, width)

            # put model into train mode
            model.train(True)

            mean_loss_epoch = 0
            for image_mask in tqdm(fine_tune_data):
                # get image and label from list
                image, label = image_mask

                # pass input data to gpu device (if cuda was available)
                image = image.to(device)
                label = label.to(device)
                
                # squeeze label batch channel dim and convert to type long
                label = label.squeeze(1).long()

                # forward images to model and get output (batch, channels, height, width)
                optimizer.zero_grad()

                # pass the input data into the model and get output
                output = model(image)

                # get loss value from tensor
                loss = criterion(output, label)

                loss_value = loss.item()
                
                mean_loss_epoch += loss_value
                #fine_tune_metrics.save_loss(epoch=epoch, loss_value=loss_value)

                # do backprogation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                optimizer.step()

                output_prob = torch.argmax(output, dim=1)

                fine_tune_metrics.update(output=output_prob, label=label)

                # chaging learing rate per interation insted of per epoch
                scheduler.step()

            # save training loss per epoch
            fine_tune_metrics.save_loss(epoch=epoch, loss_value=mean_loss_epoch/len(fine_tune_data))

            # print some info about loss and learning rate
            print(f"Epoch: {epoch}, Mean fine-tune Loss: {mean_loss_epoch/len(fine_tune_data):.8f}, Learing Rate: {optimizer.param_groups[0]['lr']:.8f}")    
            
            # print some info about metrics in training
            fine_tune_metrics_values = fine_tune_metrics.compute()

            print(f"Training Accuracy:  Class 0: {fine_tune_metrics_values['accuracy'][0]:.4f}, Class 1: {fine_tune_metrics_values['accuracy'][1]:.4f}")
            print(f"Training F1 Score:  Class 0: {fine_tune_metrics_values['f1_score'][0]:.4f}, Class 1: {fine_tune_metrics_values['f1_score'][1]:.4f}")
            print(f"Training IoU:       Class 0: {fine_tune_metrics_values['iou'][0]:.4f}, Class 1: {fine_tune_metrics_values['iou'][1]:.4f}")
            print(f"Training Precision: Class 0: {fine_tune_metrics_values['precision'][0]:.4f}, Class 1: {fine_tune_metrics_values['precision'][1]:.4f}")
            print(f"Training Recall:    Class 0: {fine_tune_metrics_values['recall'][0]:.4f}, Class 1: {fine_tune_metrics_values['recall'][1]:.4f}\n")
            
            # reset metrics for the nect epoch
            fine_tune_metrics.reset()
            
            
            # validation step
            # put model into validation mode
            model.eval()

            with torch.no_grad():

                mean_loss_epoch_val = 0
                for image_val in validation_data:
                    image, label = image_val
                    # pass input data to gpu device (if cuda is available)
                    image = image.to(device)
                    label = label.to(device)

                    # squeeze label batch channel dim and convert to type long
                    label = label.squeeze(1).long()

                    output = model(image)

                    # get loss value from tensor
                    loss = criterion(output, label)

                    loss_value = loss.item()
                    
                    mean_loss_epoch_val += loss_value
                    #validation_metrics.save_loss(epoch=epoch, loss_value=loss_value)


                    # update validation metrics
                    output_prob = torch.argmax(output, dim=1)
        
                    validation_metrics.update(output=output_prob, label=label)

                # update learning rate every epoch
                #scheduler.step(mean_loss_epoch_val/len(validation_data))

                # save validation loss per epoch 
                validation_metrics.save_loss(epoch=epoch, loss_value=mean_loss_epoch_val/len(validation_data))

                # print some info about validation loss
                print(f"Mean validation Loss: {mean_loss_epoch_val/len(validation_data):.8f}")

                # print some info about metrics in training
                validation_metrics_values = validation_metrics.compute()

                print(f"Validation Accuracy:  Class 0: {validation_metrics_values['accuracy'][0]:.4f}, Class 1: {validation_metrics_values['accuracy'][1]:.4f}")
                print(f"Validation F1 Score:  Class 0: {validation_metrics_values['f1_score'][0]:.4f}, Class 1: {validation_metrics_values['f1_score'][1]:.4f}")
                print(f"Validation IoU:       Class 0: {validation_metrics_values['iou'][0]:.4f}, Class 1: {validation_metrics_values['iou'][1]:.4f}")
                print(f"Validation Precision: Class 0: {validation_metrics_values['precision'][0]:.4f}, Class 1: {validation_metrics_values['precision'][1]:.4f}")
                print(f"Validation Recall:    Class 0: {validation_metrics_values['recall'][0]:.4f}, Class 1: {validation_metrics_values['recall'][1]:.4f}\n")
                
                # reset metric for the next epoch
                validation_metrics.reset()
                

            # save current model
            torch.save(model, f"{checkpoint_path}/fine_tune_epoch_{epoch}.pt") #

        create_fine_tune_plots(fine_tune_metrics, validation_metrics)    
        




if __name__ == "__main__":
    os.system('clear')
    # load fine-tune config
    with open('configs/seviri_configs/fine-tune_config.yaml', 'r') as file:
        fine_tune_config = yaml.safe_load(file)
    file.close()

   # load dataset config
    with open('configs/seviri_configs/dataset.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    file.close()


    # load years, months and days from config for dataset
    fine_tune_years        = dataset_config['dataset']['train']['include_years']
    fine_tune_months       = dataset_config['dataset']['train']['include_months']
    fine_tune_exclude_days = dataset_config['dataset']['train']['exclude_days']
    
    validation_years        = dataset_config['dataset']['validation']['include_years']
    validation_months       = dataset_config['dataset']['validation']['include_months']
    validation_exclude_days = dataset_config['dataset']['validation']['exclude_days']
    
    # load dataset paths
    dataset_path_images    = dataset_config['dataset']['images_path']
    dataset_path_masks     = dataset_config['dataset']['masks_path']
    data_extension         = dataset_config['dataset']['data_extension']

    # load included bands 
    include_bands          = dataset_config['dataset']['include_bands']

    # load fine-tune_config settings
    pre_trained_model_name = fine_tune_config['fine-tune']['pre_trained_model_name']
    checkpoint_path        = fine_tune_config['fine-tune']['checkpoint_save_path']
    pre_trained_model_path = fine_tune_config['fine-tune']['pre_trained_model_path']
    num_of_epochs          = fine_tune_config['fine-tune']['num_of_epochs']
    batch_size             = fine_tune_config['fine-tune']['batch_size']
    learning_rate          = fine_tune_config['fine-tune']['learning_rate']
    loss_function          = fine_tune_config['fine-tune']['loss_function']
    class_weights          = fine_tune_config['fine-tune']['class_weights']

    # load MaskedAutoEncoderViT Model h-parameters for fine-tune | Encoder h-parameters
    patch_size             = fine_tune_config['fine-tune']['patch_size']



    # fix variables types
    fine_tune_years, fine_tune_months, fine_tune_exclude_days = fix_lists(fine_tune_years, fine_tune_months, fine_tune_exclude_days)
    validation_years, validation_months, validation_exclude_days = fix_lists(validation_years, validation_months, validation_exclude_days )
    include_bands = fix_band_list(include_bands)

    patch_size = int(patch_size)
    learning_rate = float(learning_rate)
    num_of_epochs = int(num_of_epochs)
    batch_size = int(batch_size)
    

    # create output folder to save checpoints if not exists
    os.makedirs(checkpoint_path, exist_ok=True)

    # TODO: add wandb logging info
    
    # intiate pre-training
    fine_tune(
        dataset_path_images, 
        dataset_path_masks,
        data_extension, 
        fine_tune_years, 
        fine_tune_months, 
        fine_tune_exclude_days,
        validation_years,
        validation_months,
        validation_exclude_days,
        include_bands,
        checkpoint_path,
        pre_trained_model_name,
        pre_trained_model_path,
        learning_rate,
        num_of_epochs,
        batch_size,
        patch_size,
        loss_function,
        class_weights)  
