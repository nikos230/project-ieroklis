import os
import yaml
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore") # TODO: remove this in final release
from utils.utils import fix_lists, fix_band_list, get_data_files_pre_train, remove_shared_months, string_to_boolean, visualize_results, create_loss_plots
from utils.dataloader import Seviri_Dataset
from models.MaskedAutoEncoderViT.MaskedAutoEncoderViTModel import MaskedAutoEncoderViTModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def pre_train(
        dataset_path_images, 
        dataset_path_masks,
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        pre_train_exclude_days,
        validation_years,
        validation_months,
        validation_exclude_days,
        include_bands,
        checkpoint_path,
        learning_rate,
        num_of_epochs,
        batch_size,
        mask_ratio,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_pixel_loss):


        # load the data file names (paths) for pre-train data
        data_files_images_pre_train = get_data_files_pre_train(dataset_path_images, data_extension, pre_train_years, pre_train_months, pre_train_exclude_days)

        # load the data file names (paths) for validation data
        data_files_images_validation = get_data_files_pre_train(dataset_path_images, data_extension, validation_years, validation_months, validation_exclude_days)
        
        # this is a helper function, if a month from a specific year from pre-training is also present in validation data, remove it from pre-training
        data_files_images_pre_train = remove_shared_months(data_files_images_pre_train, data_files_images_validation)
        
        # define pre-train dataset | set "mode" to "pre-train" for pre-training
        pre_train_dataset = Seviri_Dataset(filenames=data_files_images_pre_train, mode='pre-train', bands=include_bands)
        print(f"Total images for pre-train: {len(pre_train_dataset)}") # TODO: enable print

        # define validation dataset | set "mode" to "pre-training" for pre-training
        validation_dataset = Seviri_Dataset(filenames=data_files_images_validation, mode='pre-train', bands=include_bands)
        print(f"Total images for validation: {len(validation_dataset)}")
        
        # TODO: fix: UserWarning: Creating a tensor from a list of numpy.ndarrays 
        # is extremely slow. Please consider converting the list to a single 
        # numpy.ndarray with numpy.array() before converting to a tensor. 

        # define dataloader and load files into ram
        pre_train_data = DataLoader(pre_train_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0)

        # define validation data and load into ram
        validation_data = DataLoader(validation_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0)                            
        

        # Seviri input data have shape of (batch, channels, height, width) = (64, 11, 32, 32)
        input_image_size      = pre_train_dataset[0].shape[1] # get input size by height of the image
        input_channels_number = pre_train_dataset[0].shape[0] # get input channels if the image
        
        # initiate model with parameters
        model = MaskedAutoEncoderViTModel(img_size=input_image_size,
                                          patch_size=patch_size,
                                          mask_ratio=mask_ratio,
                                          in_chans=input_channels_number,
                                          embed_dim=embed_dim,
                                          depth=depth,
                                          num_heads=num_heads,
                                          decoder_embed_dim=decoder_embed_dim,
                                          decoder_depth=decoder_depth,
                                          decoder_num_heads=decoder_num_heads,
                                          mlp_ratio=mlp_ratio,
                                          norm_pixel_loss=norm_pixel_loss)

        # check that gpu device is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f'Using: {device} for pre-training') # TODO: enable print

        # pass the model to GPU
        model.to(device)

        # using AdamW optimizer for ViT models
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4) # TODO: add weight decay and betas
        
        # calculate number of steps per epoch
        steps_per_epoch = len(pre_train_dataset)  # Assuming train_loader is your DataLoader
        total_steps = num_of_epochs * steps_per_epoch

        # using learing rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

        # track loss and rmse
        metrics = {}

        # main pre-train loop
        for epoch in range(num_of_epochs): # image shape = (batch, channels, height, width)
            
            # put model into train mode
            model.train(True)

            for image in tqdm(pre_train_data):
                mean_loss_interation = 0

                # pass input data to gpu device (if cuda was available)
                image = image.to(device)
                
                # forward images to model ands get loss, prediction and mask
                optimizer.zero_grad()

                #scaler = GradScaler()

                loss, prediction, mask = model(image)

                # get loss value from tensor
                loss_value = loss.item()
                
                mean_loss_interation += loss_value

                # do backprogation
                #scaler.scale(loss).backward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                optimizer.step()

                #scaler.update()

                # chaging learing rate per interation isted of per epoch
                #scheduler.step(loss_value)

                
                
            print(f"Epoch: {epoch}, Mean pre-train Loss: {mean_loss_interation/len(pre_train_dataset)}, Learing Rate: {optimizer.param_groups[0]['lr']}")    
            
            # validation step

            # put model into validation mode
            model.eval()

            with torch.no_grad():
                for image_val in validation_data:
                    mean_loss_interation_val = 0

                    # pass input data to gpu device (if cuda was available)
                    image = image.to(device)

                    loss, prediction, mask = model(image)

                    # get loss value from tensor
                    loss_value = loss.item()
                    
                    mean_loss_interation_val += loss_value

                print(f"Mean validation Loss: {mean_loss_interation_val/len(validation_dataset)}\n")

            # save average metrics for epoch
            metrics[epoch] = {}
            metrics[epoch]['pre_train_loss'] = mean_loss_interation / len(pre_train_dataset)
            metrics[epoch]['validation_loss'] = mean_loss_interation_val / len(validation_dataset)

            # save current model
            torch.save(model, f"{checkpoint_path}/epoch_{epoch}")

        visualize_results(model, validation_data)
        create_loss_plots(metrics)
        


            










if __name__ == "__main__":
    os.system('clear')
    # load pre-train config
    with open('configs/pre-train_config.yaml', 'r') as file:
        pre_train_config = yaml.safe_load(file)
    file.close()

   # load dataset config
    with open('configs/dataset.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    file.close()


    # load years, months and days from config for dataset
    pre_train_years        = dataset_config['dataset']['train']['include_years']
    pre_train_months       = dataset_config['dataset']['train']['include_months']
    pre_train_exclude_days = dataset_config['dataset']['train']['exclude_days']
    
    validation_years        = dataset_config['dataset']['validation']['include_years']
    validation_months       = dataset_config['dataset']['validation']['include_months']
    validation_exclude_days = dataset_config['dataset']['validation']['exclude_days']

    # load dataset paths
    dataset_path_images    = dataset_config['dataset']['images_path']
    dataset_path_masks     = dataset_config['dataset']['masks_path']
    data_extension         = dataset_config['dataset']['data_extension']

    # load included bands 
    include_bands          = dataset_config['dataset']['include_bands']

    # load pre-train settings
    checkpoint_path        = pre_train_config['pre-train']['checkpoint_save_path']
    num_of_epochs          = pre_train_config['pre-train']['num_of_epochs']
    batch_size             = pre_train_config['pre-train']['batch_size']
    learning_rate          = pre_train_config['pre-train']['learning_rate']
    mask_ratio             = pre_train_config['pre-train']['mask_ratio']

    # load MaskedAutoEncoderViT Model h-parameters for pre-training | Encoder h-parameters
    patch_size             = pre_train_config['pre-train']['patch_size']
    embed_dim              = pre_train_config['pre-train']['embed_dim']
    depth                  = pre_train_config['pre-train']['depth']
    num_heads              = pre_train_config['pre-train']['num_heads']

    # load MaskedAutoEncoderViT Model h-parameters for pre-training | Decoder h-parameters
    decoder_embed_dim      = pre_train_config['pre-train']['decoder_embed_dim']
    decoder_depth          = pre_train_config['pre-train']['decoder_depth']
    decoder_num_heads      = pre_train_config['pre-train']['decoder_num_heads']

    # load MaskedAutoEncoderViT Model h-parameters for pre-training | addition settings
    mlp_ratio              = pre_train_config['pre-train']['mlp_ratio']
    norm_pixel_loss        = pre_train_config['pre-train']['norm_pixel_loss']


    # fix variables types
    pre_train_years, pre_train_months, pre_train_exclude_days = fix_lists(pre_train_years, pre_train_months, pre_train_exclude_days)
    validation_years, validation_months, validation_exclude_days = fix_lists(validation_years, validation_months, validation_exclude_days )
    include_bands = fix_band_list(include_bands)

    patch_size = int(patch_size)
    learning_rate = float(learning_rate)
    num_of_epochs = int(num_of_epochs)
    batch_size = int(batch_size)
    mask_ratio = float(mask_ratio)
    embed_dim = int(embed_dim)
    depth = int(depth)
    num_heads = int(num_heads)
    decoder_embed_dim = int(decoder_embed_dim)
    decoder_depth = int(decoder_depth)
    decoder_num_heads = int(decoder_num_heads)
    mlp_ratio = int(mlp_ratio) # TODO: check if this needs to be a float insted of integer
    norm_pixel_loss = string_to_boolean(norm_pixel_loss)

    # create output folder to save checpoints if not exists
    os.makedirs(checkpoint_path, exist_ok=True)

    # TODO: add wandb logging info
    
    # intiate pre-training
    pre_train(
        dataset_path_images, 
        dataset_path_masks,
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        pre_train_exclude_days,
        validation_years,
        validation_months,
        validation_exclude_days,
        include_bands,
        checkpoint_path,
        learning_rate,
        num_of_epochs,
        batch_size,
        mask_ratio,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_pixel_loss)  
