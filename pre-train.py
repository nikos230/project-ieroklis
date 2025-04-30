import os
import yaml
from utils.utils import fix_lists, fix_band_list, get_data_files_pre_train
from utils.utils import get_data_files_fine_tune # remove this, just for test
from utils.dataloader import Seviri_Dataset
from models.MaskedAutoEncoderViT import MaskedAutoEncoderModel



def pre_train(
            dataset_path_images, 
            dataset_path_masks,
            data_extension, 
            pre_train_years, 
            pre_train_months, 
            pre_train_exclude_days,
            include_bands):


        # load the data file names
        data_files_images = get_data_files_pre_train(dataset_path_images, data_extension, pre_train_years, pre_train_months, pre_train_exclude_days)

        # define pre-train dataset | set "mode" to "pre-train" for pre-training
        pre_train_dataset = Seviri_Dataset(filenames=data_files_images, mode='pre-train', bands=include_bands)
       
        #print(pre_train_dataset[1].shape) #  pre_train_dataset[index of sample][ index of channel]

        model = MaskedAutoEncoderModel()







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
    pre_train_years        = dataset_config['dataset']['include_years']
    pre_train_months       = dataset_config['dataset']['include_months']
    pre_train_exclude_days = dataset_config['dataset']['exclude_days']

    # load dataset paths
    dataset_path_images    = dataset_config['dataset']['images_path']
    dataset_path_masks     = dataset_config['dataset']['masks_path']
    data_extension         = dataset_config['dataset']['data_extension']

    # load included bands 
    include_bands          = dataset_config['dataset']['include_bands']

    # load pre-train settings
    


    pre_train_years, pre_train_months, pre_train_exclude_days = fix_lists(pre_train_years, pre_train_months, pre_train_exclude_days)
    include_bands = fix_band_list(include_bands)

   

    pre_train(
        dataset_path_images, 
        dataset_path_masks,
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        pre_train_exclude_days,
        include_bands)  
