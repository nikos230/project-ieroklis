import os
from xarray.core.groupby import _validate_groupby_squeeze
import yaml
import torch
import torch.nn as nn
import csv
import wandb
from utils.utils import get_years_months, get_files_MedST28, string_to_boolean, calc_params
from utils.MedST28_dataloader import MedST28_Dataset
from models.MaskedAutoEncoderViT.MaskedAutoEncoderViTModel import MaskedAutoEncoderViTModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def pre_train(
        dataset_path,
        path_to_MedST28_stats, 
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        validation_years,
        validation_months,
        include_variables,
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
        norm_pixel_loss,
        time_steps,
        samples_patch_size,
        reference_date,
        visualize_results_path):


        # load the data file names (paths) for pre-train data
        data_files_pre_train = get_files_MedST28(dataset_path, data_extension, pre_train_years, pre_train_months)

        # load the data file names (paths) for validation data
        data_files_validation = get_files_MedST28(dataset_path, data_extension, validation_years, validation_months)
        
        # define pre-train dataset | set "mode" to "pre-train" for pre-training
        pre_train_dataset = MedST28_Dataset(filenames=data_files_pre_train, 
                                            mode='pre-train',
                                            training_data=True, 
                                            variables=include_variables, 
                                            time_steps=time_steps, 
                                            stats_path=path_to_MedST28_stats, 
                                            reference_date=reference_date,
                                            samples_patch_size=samples_patch_size)
        
        print(f"Total samples for pre-train: {len(pre_train_dataset)}") 

        # define validation dataset | set "mode" to "pre-training" for pre-training
        validation_dataset = MedST28_Dataset(filenames=data_files_validation, 
                                             mode='pre-train',
                                             training_data=False,  
                                             variables=include_variables, 
                                             time_steps=time_steps, 
                                             stats_path=path_to_MedST28_stats, 
                                             reference_date=reference_date,
                                             samples_patch_size=samples_patch_size)
        
        print(f"Total samples for validation: {len(validation_dataset)}")
        

        # define dataloader and load files into ram
        pre_train_data = DataLoader(pre_train_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4)

        # define validation data and load into ram
        validation_data = DataLoader(validation_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4)                            
        

        # Seviri input data have shape of (batch, channels, height, width) = (64, 11, 32, 32)
        input_image_size      = pre_train_dataset[0].shape[1] # get input size by height of the image
        input_channels_number = pre_train_dataset[0].shape[0] # get input channels 
        print(f"Input image size: {input_image_size}, Channels number: {input_channels_number}")

        

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
        
        # calculate number of parameters
        print(calc_params(input_image_size, input_channels_number, patch_size, embed_dim, depth, mlp_ratio))
        # exit()
        # check that gpu device is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pass the model to GPU
        model.to(device)

        # using AdamW optimizer for ViT models
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=1e-4)
        
        # calculate number of steps per epoch
        steps_per_epoch = len(pre_train_dataset)  # Assuming train_loader is your DataLoader
        total_steps = num_of_epochs * steps_per_epoch

        # using learing rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        # track loss 
        metrics = {}

        # main pre-train loop
        for epoch in range(num_of_epochs): # image shape = (batch, channels, height, width)
            
            # put model into train mode
            model.train()

            mean_loss_interation = 0
            for variables in tqdm(pre_train_data):
                # pass input data to gpu device (if cuda is available)
                variables = variables.to(device)
                
                # forward images to model ands get loss, prediction and mask
                optimizer.zero_grad()

                loss, prediction, mask = model(variables)
                
                # get loss value from tensor
                loss_value = loss.item()
                
                mean_loss_interation += loss_value

                # do backprogation
                #scaler.scale(loss).backward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                optimizer.step()

                # changing learning rate per interation insted of per epoch
                scheduler.step()

            # print loss info
            print(f"Epoch: {epoch}, Mean pre-train Loss: {mean_loss_interation/len(pre_train_dataset)}, Learing Rate: {optimizer.param_groups[0]['lr']}")    
            

            # validation step
            # put model into validation mode
            model.eval()

            mean_loss_interation_val = 0
            with torch.no_grad():
                for variables in validation_data:
                    # pass input data to gpu device (if cuda is available)
                    variables = variables.to(device)

                    loss, prediction, mask = model(variables)

                    # get loss value from tensor
                    loss_value = loss.item()
                    
                    mean_loss_interation_val += loss_value
                
                # update learning rate every epoch
                #scheduler.step(mean_loss_interation_val/len(validation_data))

                # print some info
                print(f"Mean validation Loss: {mean_loss_interation_val/len(validation_data)}\n")

                # log pre-train and validation loss in wandb
                run.log({'pre-train loss': mean_loss_interation/len(pre_train_dataset), 
                         'validation loss': mean_loss_interation_val/len(validation_data)},
                         step=epoch)

            # save average metrics for epoch
            metrics[epoch] = {}
            metrics[epoch]['pre_train_loss'] = mean_loss_interation / len(pre_train_data)
            metrics[epoch]['validation_loss'] = mean_loss_interation_val / len(validation_data)
            
            # save current model
            torch.save(model, f"{checkpoint_path}/pre_train_epoch_{epoch}.pt")

        # close wandb logging
        run.finish()

        # save loss metrics to .csv file (for later visualization)
        with open(f"{visualize_results_path}/loss_values.csv", 'w', newline="\n") as loss_file:
             writer = csv.writer(loss_file)
             writer.writerow(['epoch', 'pre_train_loss', 'validation_loss'])

             for epoch, loss in metrics.items():
                  writer.writerow([epoch, loss['pre_train_loss'], loss['validation_loss']])
        
    

        
        
        
       
        



if __name__ == "__main__":
    os.system('clear')
    # load pre-train config
    with open('configs/MedST-28_configs/pre-train_config.yaml', 'r') as file:
        pre_train_config = yaml.safe_load(file)
    file.close()

   # load dataset config
    with open('configs/MedST-28_configs/MedST28_dataset.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    file.close()

    
    # load years, months and days from config for dataset
    pre_train_years        = dataset_config['dataset']['train']['include_years']
    pre_train_months       = dataset_config['dataset']['train']['include_months']
    
    validation_years        = dataset_config['dataset']['validation']['include_years']
    validation_months       = dataset_config['dataset']['validation']['include_months']

    test_years              = dataset_config['dataset']['test']['include_years']
    test_months             = dataset_config['dataset']['test']['include_months']

    # load dataset paths
    dataset_path           = dataset_config['dataset']['path_to_dataset']
    path_to_MedST28_stats  = dataset_config['dataset']['path_to_stats']
    data_extension         = dataset_config['dataset']['data_extension']
    time_steps             = dataset_config['dataset']['time_steps']
    samples_patch_size     = dataset_config['dataset']['samples_patch_size']
    reference_date         = dataset_config['dataset']['reference_date']

    # load included variables 
    include_variables      = dataset_config['dataset']['include_variables']

    # load pre-train settings
    checkpoint_path        = pre_train_config['pre-train']['checkpoint_save_path']
    num_of_epochs          = pre_train_config['pre-train']['num_of_epochs']
    batch_size             = pre_train_config['pre-train']['batch_size']
    learning_rate          = pre_train_config['pre-train']['learning_rate']
    mask_ratio             = pre_train_config['pre-train']['mask_ratio']
    visualize_results_path = pre_train_config['pre-train']['visualize_results_path']

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
    pre_train_years, pre_train_months = get_years_months(pre_train_years, pre_train_months, dataset_path)
    validation_years, validation_months = get_years_months(validation_years, validation_months, dataset_path)
    test_years, test_months = get_years_months(test_years, test_months, dataset_path)

    # remove shared years between pre-training and validation and test years
    pre_train_years = list(set(pre_train_years) - set(validation_years) - set(test_years))


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
    mlp_ratio = int(mlp_ratio) # TODO: check if this needs to be a float instead of integer
    norm_pixel_loss = string_to_boolean(norm_pixel_loss)

    # create output folder to save checkpoints if not exist
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(visualize_results_path, exist_ok=True)

    # wandb log info
    run = wandb.init(
         project="project-ieroklis"
    )

    # initiate pre-training
    pre_train(
        dataset_path,
        path_to_MedST28_stats, 
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        validation_years,
        validation_months,
        include_variables,
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
        norm_pixel_loss,
        time_steps,
        samples_patch_size,
        reference_date,
        visualize_results_path)  
        
        
        