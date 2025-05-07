import os
import glob
import matplotlib.pyplot as plt
import torch


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


def visualize_results(model, pre_train_data):
    print(f"Creating pre-training plot results...")

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





    os.makedirs('output', exist_ok=True)
    plt.tight_layout()
    plt.suptitle("Pre-training Results", fontsize=14)
    plt.savefig('output/pre-train_results.png', dpi=600, bbox_inches='tight')
    plt.close()    
    print(f'Done!')
    
    return 0


# a function to generate loss (RMSE) plots
def create_loss_plots(metrics):

    # get list with epochs
    epochs = list(metrics.keys())
    
    # extract loss values from metrics dictinary
    pre_train_loss = [metrics[epoch]['pre_train_loss'] for epoch in epochs]
    validation_loss = [metrics[epoch]['validation_loss'] for epoch in epochs]
    
    # plot pre-train loss
    plt.plot(epochs, pre_train_loss, label='pre-train loss', color='red', marker='o')
    
    # plot validation loss
    plt.plot(epochs, validation_loss, label='validation loss', color='blue', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.title('Pre-train and Validation Loss')
    plt.tight_layout()
    plt.savefig('output/train_valdation_loss', dpi=600)
    plt.close()
    

            




