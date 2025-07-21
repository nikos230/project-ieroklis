import os
import glob
import matplotlib.pyplot as plt
from timm.data import dataset
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoConfig


def check_valid_date(data_file, pre_train_years, pre_train_months, pre_train_exclude_days):
    valid_year  = False
    valid_month = False
    valid_day   = False


    # extract the date from file name
    file_name = os.path.basename(data_file)
    
    # extract the timedate string from the filename
    timedate = file_name.split('_')[0]

    # extract the year from the timedate string
    year = str(timedate[0] + timedate[1] + timedate[2] + timedate[3])

    # extract the month from timedate string
    month = str(timedate[4] + timedate[5])

    # extract the day fro timedate string
    day = str(timedate[6] + timedate[7])

    # check if data_file is within the required years, months and days
    if pre_train_years == -1:
        valid_year = True
    else:
        if year in pre_train_years:
            valid_year = True
        else:
            valid_year = False

    if pre_train_months == -1:
        valid_month = True
    else:
        if month in pre_train_months:
            valid_month = True
        else:
            valid_month = False  

     
    if pre_train_exclude_days == -1:
        valid_day = True
    else:
        if day in pre_train_exclude_days:
            valid_day = False
        else:
            valid_day = True        
                  
    # finally return True if all check are good else return False to not include this image
    if not valid_year or not valid_month or not valid_day:
        return False
    else:
        return True    



def get_unique_numbers(filename):
    unique_number = 0
    patch_number  = 0

    # extract the data from file name
    file_name = os.path.basename(filename)
    file_name_1 = file_name.split('_')[0]
    file_name_2 = file_name.split('_')[2].split('.')[0]
    
    # get uniquie number from filename
    unique_number = str(file_name_1[7] + file_name_1[8] + file_name_1[9] + file_name_1[10] + file_name_1[11] + file_name_1[12] + file_name_1[13])

    # get patch numbe from file name
    patch_number = file_name_2
    
    return unique_number, patch_number



def get_data_files_pre_train(dataset_path_images, data_extension, pre_train_years, pre_train_months, pre_train_exclude_days):
    # get a list containing all file names of the images (from images folder)
    data_files = glob.glob(dataset_path_images + '/*' + data_extension)

    # check date for all files and exclude some if needed
    data_files_DateValidated = []

    for data_file in data_files:
        if check_valid_date(data_file, pre_train_years, pre_train_months, pre_train_exclude_days) == True:
            data_files_DateValidated.append(data_file)
        else:
            continue         

    return data_files_DateValidated



def get_data_files_fine_tune(dataset_path_images, dataset_path_masks, data_extension, pre_train_years, pre_train_months, pre_train_exclude_days):
    data_list = []

    # get a list containing all file names of the images (from images folder)
    data_files_images = glob.glob(dataset_path_images + '/*' + data_extension)
    data_files_masks  = glob.glob(dataset_path_masks + '/*' + data_extension)

    # check date for all files and exclude some if needed
    data_files_images_DateValidated = []

    for data_file in data_files_images:
        if check_valid_date(data_file, pre_train_years, pre_train_months, pre_train_exclude_days) == True:
            data_files_images_DateValidated.append(data_file)
        else:
            continue        
    

    for file_image in data_files_images_DateValidated:
        image_file_name = file_image
        unique_number_image, patch_number_image = get_unique_numbers(file_image)

        # search for correct mask for every image file
        for file_mask in data_files_masks:
            unique_number_mask, patch_number_mask = get_unique_numbers(file_mask)

            if unique_number_image == unique_number_mask and patch_number_image == patch_number_mask:
                mask_file_name = file_mask

                # add image and mask filenames to data list
                data_list.append((image_file_name, mask_file_name))
                break
            else:
                continue    

    return data_list


# remove any shared images in pre-trained set and validation or test set, from pre-train set
def remove_shared_months(pre_train_data_files, validation_data_files):
    validation_set = set(validation_data_files)
    return [image for image in pre_train_data_files if image not in validation_set]
        

# function to fix years, months and days lists from yaml config file
def fix_lists(pre_train_years, pre_train_months, pre_train_exlude_days):
    if pre_train_years == "all":
        pre_train_years = -1
    else:    
        pre_train_years = [pre_train_years]
        pre_train_years = pre_train_years[0].split(', ')

    if pre_train_months == "all":
        pre_train_months = -1
    else:
        pre_train_months = [pre_train_months]
        pre_train_months = pre_train_months[0].split(', ')
        pre_train_months = [f"0{num}" if len(str(num)) == 1 else str(num) for num in pre_train_months]

    if pre_train_exlude_days == "None":
        pre_train_exlude_days = -1
    else:
        pre_train_exlude_days = [pre_train_exlude_days]
        pre_train_exlude_days = pre_train_exlude_days[0].split(', ')   
        pre_train_exlude_days = [f"0{num}" if len(str(num)) == 1 else str(num) for num in pre_train_exlude_days]


    return pre_train_years, pre_train_months, pre_train_exlude_days


# bands are in string form, covert them into integers
def fix_band_list(bands):
    bands = [bands] 
    bands = bands[0].split(', ')
    bands = list(map(int, bands))
    
    return bands


# function to turn "True" or "False" to True and False (string to boolean)
def string_to_boolean(string):
    if string == "False":
        return False
    else:
        return True    


def visualize_results(model, pre_train_data, temporal=None):
    print(f"Creating pre-training plot results...")

    if temporal == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        counter = 0

        preds_and_masks = []
        
        for batch in pre_train_data:
            for image in batch:
                
                image = image.to(device)
                
                loss, prediction, mask = model(image.unsqueeze(0))       

                patch_size = 4
                img_size = 32
                num_patches_per_row = img_size // patch_size
                B = 1
                C, N, D = image.shape

                # reshape of prediction images
                prediction = prediction.view(B, num_patches_per_row, num_patches_per_row, patch_size, patch_size, C)
                prediction = prediction.permute(0, 5, 1, 3, 2, 4)
                prediction = prediction.contiguous().view(B, C, img_size, img_size)

                # reshape of masks
                mask = mask.view(B, 8, 8, 1, 1, 1)
                mask = mask.expand(-1, -1, -1, patch_size, patch_size, 1)
                mask = mask.permute(0, 5, 1, 3, 2, 4).contiguous()
                mask = mask.view(B, 1, img_size, img_size)

                # remove batch dimention
                prediction = prediction.squeeze(0)
                mask = mask.squeeze(0)

                # append into a list the image, prediction and the mask
                preds_and_masks.append((image, prediction, mask))
            
        # make a plot
        rows = 2 # 2
        cols = 11 # 4

        fig, axis = plt.subplots(rows, cols, figsize=(20, 5.5))
        #axis = axis.flatten()

        image_, prediction_, mask_ = preds_and_masks[0]

        for k in range(0, 1):
            for i in range(0, image.shape[0]):

                image_, prediction_, mask_ = preds_and_masks[k]

            
                # plot 1st channel | top image | bottom prediction
                axis[0, i].imshow(image_[i].cpu().detach().numpy())
                axis[0, i].set_title(f"Channel {i+1}")
                axis[0, i].axis('off')

                axis[1, i].imshow(prediction_[i].cpu().detach().numpy())
                axis[1, i].set_title(f"Reconstracted\nChannel {i+1}")
                axis[1, i].axis('off')

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        counter = 0

        preds_and_masks = []

        
        for variables, timestamps in pre_train_data:
            
            variables = variables[0:1]  # keep batch dimension (batch size = 1)
            timestamps = timestamps[0:1]

            variables = variables.to(device)
            timestamps = timestamps.to(device)
  
            loss, prediction, mask = model(variables, timestamps)     
            
            patch_size = 8
            img_size = 128
            num_patches_per_row = img_size // patch_size
            num_patches_per_img = num_patches_per_row ** 2
            B = 1
            T = 30
            image = variables[0][0] 
            C, N, D = image.shape

            # reshape of prediction images
            prediction = prediction.view(B, T, num_patches_per_img, patch_size, patch_size, C)
            prediction = prediction.permute(0, 1, 5, 2, 3, 4)
            prediction = prediction.contiguous().view(B, T, C, num_patches_per_row, patch_size, num_patches_per_row, patch_size)
            prediction = prediction.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
            prediction = prediction.view(B, T, C, img_size, img_size)

            # reshape of masks
            mask = mask.view(B, T, num_patches_per_row, num_patches_per_row, 1, 1, 1)
            mask = mask.expand(-1, -1, -1, -1, patch_size, patch_size, 1)
            mask = mask.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
            mask = mask.view(B, T, 1, img_size, img_size)

            # remove batch dimension
            prediction = prediction.squeeze(0)
            mask = mask.squeeze(0)

            # append into a list the image, prediction and the mask
            preds_and_masks.append((variables, prediction, mask))
        
    # make a plot
    rows = 5 # 2
    cols = 8 # 4

    #fig, axis = plt.subplots(rows, cols, figsize=(20, 5.5))
    fig, axis = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 3))
    #axis = axis.flatten()

    image_, prediction_, mask_ = preds_and_masks[0]

    # select a variable to plot from the 29 variables
    image_plot = image_[0][0]
    prediction_plot = prediction_[0]
    mask_plot = mask_[0][0]

    num_channels = 29
    cols = 8
    rows = (num_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2, rows * 4))

    # Flatten axes array for easy indexing
    axes = axes.flatten()


    for i in range(num_channels):
        # Original image on top row (even indices)
        axes[2*i].imshow(image_plot[i].cpu().detach().numpy(), cmap='gray')
        axes[2*i].set_title(f"Channel {i+1}", fontsize=10)
        axes[2*i].axis('off')

        # Reconstructed prediction below (odd indices)
        axes[2*i + 1].imshow(prediction_plot[i].cpu().detach().numpy(), cmap='gray')
        axes[2*i + 1].set_title(f"Reconstructed\nChannel {i+1}", fontsize=10)
        axes[2*i + 1].axis('off')

    # Hide any unused subplots (if total axes > needed)
    for j in range(2 * num_channels, len(axes)):
        axes[j].axis('off')




    os.makedirs('output', exist_ok=True)
    plt.tight_layout()
    plt.suptitle("Pre-training Results", fontsize=10)
    plt.savefig('output/pre-train_results_MedST28.png', dpi=600, bbox_inches='tight')
    plt.close('all')    
    print(f'Done!')
    


# a function to generate loss (RMSE) plots
def create_loss_plots(metrics):

    # get list with epochs
    epochs = list(metrics.keys())
    
    # extract loss values from metrics dictinary
    pre_train_loss = [metrics[epoch]['pre_train_loss'] for epoch in epochs]
    validation_loss = [metrics[epoch]['validation_loss'] for epoch in epochs]
    
    # plot pre-train loss
    plt.plot(epochs, pre_train_loss, label='pre-train loss', color='red', marker='')
    
    # plot validation loss
    plt.plot(epochs, validation_loss, label='validation loss', color='blue', marker='')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.title('Pre-train and Validation Loss')
    plt.tight_layout()
    plt.savefig('output/train_valdation_loss.png', dpi=600)
    plt.close('all')



def get_configuration(model):
    # dictionary to save model config
    settings = {}
    
    # patch embeding settings
    patch_embed_config = model.patch_embed.proj
    settings['input_channels'] = patch_embed_config.in_channels
    
    # encoder blocks settings
    block = model.blocks[0]
    settings['depth'] = len(model.blocks)
    settings['embed_dim'] = patch_embed_config.out_channels
    settings['mlp_hidden_dim'] = block.mlp.fc1.out_features
    settings['mlp_ratio'] = settings['mlp_hidden_dim'] // settings['embed_dim']
    
    # patch size from pre-trained model
    settings['patch_size'] = model.patch_embed.proj.kernel_size[0]

    # attention head settings
    settings['num_heads'] = block.attn.num_heads  

    #print(settings['num_heads'])
    #print(vars(model))
    return settings


def get_configuration_Hugging_Face(model_name):
    # dictionary to save model config
    settings = {}

    # load config from Hugging Face
    model_config = AutoConfig.from_pretrained(model_name)
    print(model_config)

    # encoder blocks settings
    settings['depth'] = model_config.num_hidden_layers
    settings['embed_dim'] = model_config.hidden_size
    settings['mlp_hidden_dim'] = 3072
    settings['mlp_ratio'] = 4
 
    # attention head settings
    settings['num_heads'] = model_config.num_attention_heads  


    return settings
    



def create_fine_tune_plots(fine_tune_metrics, validation_metrics):

    fine_tune_loss = fine_tune_metrics.loss_values
    validation_loss = validation_metrics.loss_values

    # get list with epochs
    epochs = list(fine_tune_metrics.loss_values.keys())

    # extract loss values from metrics dictinary
    fine_tune_loss = [fine_tune_loss[epoch] for epoch in epochs]
    validation_loss = [validation_loss[epoch] for epoch in epochs]
    
    # plot pre-train loss
    plt.plot(epochs, fine_tune_loss, label='pre-train loss', color='red', marker='')
    
    # plot validation loss
    plt.plot(epochs, validation_loss, label='validation loss', color='blue', marker='')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.title('Fine-tune and Validation Loss')
    plt.tight_layout()
    plt.savefig('output/fine_tune_valdation_loss.png', dpi=600)
    plt.close()



def modify_state_dict_input_channels(state_dict, new_input_channels, pre_trained_input_channels):
    old_weight = state_dict['patch_embed.proj.weight']

    # initialize a new channel with zero weights if needed else do not change anything
     # input channles are greater than pre-trained model channels
    if new_input_channels > old_weight.shape[1]: 
        new_weight = torch.zeros((old_weight.shape[0], new_input_channels, old_weight.shape[2], old_weight.shape[3]))
        new_weight[:, :old_weight.shape[1], :, :] = old_weight
        new_weight[:, 10:, :, :] = old_weight.mean(dim=1, keepdim=True)
        state_dict['patch_embed.proj.weight'] = new_weight

        return state_dict
    
    # input channels are fewer than pre-trained model channels
    elif new_input_channels < old_weight.shape[1]: 
        #new_weight = old_weight[:, :new_input_channels, :, :]
        new_weight = old_weight[:, 1:, :, :]
        #print(new_weight.shape)
        #exit()
        state_dict['patch_embed.proj.weight'] = new_weight

        return state_dict

    else:
        return state_dict
        
        


def get_years_months(years, months, dataset_path):
    all_years = os.listdir(dataset_path)
    all_months = os.listdir(os.path.join(dataset_path, all_years[0]))
    
    if years == 'all':
        years = all_years

    else:
        years = [years]
        years = years[0].split(', ')

    if months == 'all':
        months = all_months

    else:
        months = [months]
        months = months[0].split(', ')  
        months = [f"0{num}" if len(str(num)) == 1 else str(num) for num in months] 

    return years, months


def get_years_months_fire_spread(years, dataset_path):
    all_years = os.listdir(dataset_path)
    #all_months = os.listdir(os.path.join(dataset_path, all_years[0]))

    # check if all items are years
    for item in all_years:
        if len(item) != 4:
            all_years.remove(item)

    if years == 'all':
        years = all_years

    else:
        years = [years]
        years = years[0].split(', ')

    return years



def get_files_MedST28(dataset_path, data_extension, years, months):
    file_names = []    

    for year in years:
        path_to_year = os.path.join(dataset_path, year)
        for month in months:
            path_to_month = os.path.join(path_to_year, month)

            files = glob.glob(f"{path_to_month}/*.nc")
            file_names.extend(files)

    return file_names        


def get_files_WildfireSpread(dataset_path, data_extension, years):
    file_names = []    

    for year in years:
        path_to_year = os.path.join(dataset_path, year)
        
        for country in os.listdir(path_to_year):
            path_to_country = os.path.join(path_to_year, country)

            files = glob.glob(f"{path_to_country}/*.nc")
            file_names.extend(files)

    return file_names


def get_files_FireRisk(dataset_path, data_extension, years):
    file_names = []    

    for year in years:
        path_to_year = os.path.join(dataset_path, year)

        files = glob.glob(f"{path_to_year}/*.nc")
        file_names.extend(files)

    return file_names




def get_timestamps_from_reference(timestamps, reference_date):
    reference_date = pd.Timestamp(reference_date)
    print(reference_date)
    print(timestamps)

    timestamps_from_reference = []
    
    for timestamp in timestamps:
        date = pd.to_datetime(timestamp)
        reference_timestamp = (date - reference_date).days
        timestamps_from_reference.append(reference_timestamp)

    return timestamps_from_reference    



def calc_params(input_image_size, input_channels_number, patch_size, embed_dim, depth, mlp_ratio):

    
    patch_embeding = ( (patch_size ** 2) * (input_channels_number * embed_dim) ) + embed_dim

    each_projection = embed_dim * embed_dim
    projections = each_projection * 3

    block_params = projections + each_projection

    mlp_size = mlp_ratio * embed_dim

    mlp_layer_1 = embed_dim * mlp_size
    mlp_layer_2 = mlp_size * embed_dim
    mlp_biases = mlp_size + embed_dim

    mlp_params = mlp_layer_1 + mlp_layer_2 + mlp_biases

    layer_norm_params = 2 * embed_dim * 2

    total_params_per_block = block_params + mlp_params + layer_norm_params

    total_params = depth * total_params_per_block

    return total_params
    