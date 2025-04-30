import os
import yaml
from utils.utils import fix_lists




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

    # load fine-tune config
    with open('configs/fine-tune_config.yaml', 'r') as file:
        fine_tune_config = yaml.safe_load(file)
    file.close()

    # load years, months and days from config for dataset
    pre_train_years        = dataset_config['dataset']['include_years']
    pre_train_months       = dataset_config['dataset']['include_months']
    pre_train_exlude_days  = dataset_config['dataset']['exclude_days']

    # load dataset path
    dataset_path_images    = dataset_config['dataset']['images_path']
    dataset_path_masks     = dataset_config['dataset']['masks_path']


    pre_train_years, pre_train_months, pre_train_exlude_days = fix_lists(pre_train_years, pre_train_months, pre_train_exlude_days)

   

    print(pre_train_years, pre_train_months,pre_train_exlude_days)    
