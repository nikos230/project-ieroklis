import os
from timm.data import dataset
from xarray import Variable
from xarray.computation.arithmetic import VariableArithmetic
from xarray.core import variable
import yaml
import torch
import torch.nn as nn
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import get_years_months, get_files_MedST28, remove_shared_months, string_to_boolean, visualize_results, create_loss_plots, get_timestamps_from_reference, get_configuration
from utils.MedST28_dataloader_temporal import MedST28_Dataset
from models.MaskedAutoEncoderViT.MaskedAutoEncoderViTModelTemporal import MaskedAutoEncoderViTModelTemporal
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def pre_train(
        dataset_path,
        path_to_MedST28_stats, 
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        validation_years,
        validation_months,
        include_variables,
        time_steps,
        reference_date,
        visualize_results_path,
        pre_trained_model_path):


        # load the data file names (paths) for pre-train data
        data_files_pre_train = get_files_MedST28(dataset_path, data_extension, pre_train_years, pre_train_months)

        # load the data file names (paths) for validation data
        data_files_validation = get_files_MedST28(dataset_path, data_extension, validation_years, validation_months)
        
        # define pre-train dataset | set "mode" to "pre-train" for pre-training
        pre_train_dataset = MedST28_Dataset(filenames=data_files_pre_train, 
                                            mode='pre-train', 
                                            variables=include_variables, 
                                            time_steps=time_steps, 
                                            stats_path=path_to_MedST28_stats, 
                                            reference_date=reference_date)
        
        print(f"Total samples for pre-train: {len(pre_train_dataset)}") 

        # define validation dataset | set "mode" to "pre-train" for pre-training
        validation_dataset = MedST28_Dataset(filenames=data_files_validation, 
                                             mode='pre-train', 
                                             variables=include_variables, 
                                             time_steps=time_steps, 
                                             stats_path=path_to_MedST28_stats, 
                                             reference_date=reference_date)
        
        print(f"Total samples for validation: {len(validation_dataset)}")
        

        # define dataloader and load files into ram
        pre_train_data = DataLoader(pre_train_dataset, 
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=2)

        # define validation data and load into ram
        validation_data = DataLoader(validation_dataset, 
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=2)                            
        

        # Seviri input data have shape of (batch, channels, height, width) = (64, 11, 32, 32)
        input_image_size      = pre_train_dataset[0][0].shape[2] # get input size by height of the image
        input_channels_number = pre_train_dataset[0][0].shape[1] # get input channels 
        input_time_steps_number = pre_train_dataset[0][0].shape[0] # get input time steps 
        print(f"Input image size: {input_image_size}, Channels number: {input_channels_number}, Time Steps: {input_time_steps_number}")

        # define device CPU
        device = 'cpu'

        model = torch.load(pre_trained_model_path, weights_only=False)

        # pass the model to CPU
        model.to(device)
        model.eval()

        # model settings
        model_config = get_configuration(model)

        with torch.no_grad():
            for variables, timestamps in validation_data:

                # pass input data to gpu device (if cuda is available)
                variables = variables.to(device)

                # pass timestamps into gpu device (if cuda is available)
                timestamps = timestamps.to(device)

                loss, prediction, mask = model(variables, timestamps)
                
                prediction = reshape(prediction,
                                     model_config['patch_size'],
                                     input_image_size,
                                     input_time_steps_number,
                                     input_channels_number,
                                     item='prediction')

                mask = reshape(mask,
                               model_config['patch_size'],
                               input_image_size,
                               input_time_steps_number,
                               input_channels_number,
                               item='mask')
                
                
                plot_loss(visualize_results_path)
                visualize_pre_train_reconstruct(visualize_results_path, prediction, mask, variables, include_variables)
                exit()
                



def visualize_pre_train_reconstruct(visualize_results_path, predictions, masks, variables, variable_names):
    # remove bach dimension from predictions, masks and variables
    predictions = predictions.squeeze(0)
    masks = masks.squeeze(0)
    variables = variables.squeeze(0)
    
    # select a time step
    time_step = 0

    # get the time step for all variables
    predictions = predictions[0]
    masks = masks[0]
    variables = variables[0]

    # plot the predictions and the variables as well as the masks
    num_vars = predictions.shape[0]
    max_cols = 5
    num_rows = 2 * (int(num_vars / max_cols) + 1)
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols*5, num_rows*4), squeeze=True)
    variable_index = 0

    for row in range(0, num_rows - 1, 2):
        for col in range(0, max_cols):
            if variable_index >= num_vars:
                break
            
            # plot original variable
            axes[row, col].imshow(variables[variable_index], cmap='gray')
            axes[row, col].set_title(f"{variable_names[variable_index]}", fontsize=14)

            # plot model predictions (reconstructed variables)
            axes[row + 1, col].imshow(predictions[variable_index], cmap='gray')
            axes[row + 1, col].set_title(f"Reconstructed {variable_names[variable_index]}", fontsize=14)

            # plot masks
            #axes[row + 1, col].imshow(masks[0])

            variable_index += 1

    plt.tight_layout()
    plt.savefig(f"{visualize_results_path}/reconstruct_pre_train.png", dpi=300)
    plt.close('all')
    print(f"Saved reconstructed variables in {visualize_results_path}")
     


def plot_loss(path_to_visualize_results_path):
    loss_values = pd.read_csv(f"{path_to_visualize_results_path}/loss_values.csv")
    
    # create the loss plot
    plt.plot(loss_values['epoch'], loss_values['pre_train_loss'], label='Pre-train Loss', color='blue')
    plt.plot(loss_values['epoch'], loss_values['validation_loss'], label='Validation Loss', color='red')

    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.title('Pre-train and Validation Loss')
    plt.legend()
    plt.savefig(f"{path_to_visualize_results_path}/pre_train_loss_plot.png", dpi=600)
    plt.close('all')
    print(f"Loss diagram saved at {path_to_visualize_results_path}")

        
        
def reshape(output, patch_size, image_size, time_steps_num, channels_num, item=None):
    # output can be the prediction of the MAE ViT Model or the mask
    B = 1
    num_paches_per_row = image_size // patch_size
    num_patches_per_img = num_paches_per_row ** 2
     
    if item == 'prediction':
        output = output.view(B, time_steps_num, num_patches_per_img, patch_size, patch_size, channels_num)  
        output = output.permute(0, 1, 5, 2, 3, 4)
        output = output.contiguous().view(B, time_steps_num, channels_num, num_paches_per_row, patch_size, num_paches_per_row, patch_size)
        output = output.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        output = output.view(B, time_steps_num, channels_num, image_size, image_size)

        return output
    
    elif item == 'mask':
        output = output.view(B, time_steps_num, num_paches_per_row, num_paches_per_row, 1, 1, 1)
        output = output.expand(-1, -1, -1, -1, patch_size, patch_size, 1)
        output = output.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
        output = output.view(B, time_steps_num, 1, image_size, image_size)    

        return output
    
    else:
         return 0

        



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

    # with open('configs/MedST-28_configs/fine-tune_config.yaml', 'r') as file:
    #      fine_tune_config = yaml.safe_load(file)
    # file.close()     

    
    # load pre-trained model path
    #pre_trained_model_path = fine_tune_config['fine-tune']['pre_trained_model_path']
    pre_trained_model_path = 'pre-trained_models/MedST28_test_2.pt'

    # load years, months and days from config for dataset
    pre_train_years        = dataset_config['dataset']['train']['include_years']
    pre_train_months       = dataset_config['dataset']['train']['include_months']
    
    validation_years        = dataset_config['dataset']['validation']['include_years']
    validation_months       = dataset_config['dataset']['validation']['include_months']

    # load dataset paths
    dataset_path           = dataset_config['dataset']['path_to_MedST28_dataset']
    path_to_MedST28_stats  = dataset_config['dataset']['path_to_MedST28_stats']
    data_extension         = dataset_config['dataset']['data_extension']
    time_steps             = dataset_config['dataset']['time_steps']
    reference_date         = dataset_config['dataset']['reference_date']

    # load included variables 
    include_variables      = dataset_config['dataset']['include_variables']

    visualize_results_path = pre_train_config['pre-train']['visualize_results_path']

    

    # fix variables types
    pre_train_years, pre_train_months = get_years_months(pre_train_years, pre_train_months, dataset_path)
    validation_years, validation_months = get_years_months(validation_years, validation_months, dataset_path)

    # remove shared years between pre-training and validation
    pre_train_years = list(set(pre_train_years) - set(validation_years))


    # create output folder to save results if not exist
    os.makedirs(visualize_results_path, exist_ok=True)


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
        time_steps,
        reference_date,
        visualize_results_path,
        pre_trained_model_path)  
        
        
        