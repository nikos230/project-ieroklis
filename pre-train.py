import os
import yaml
import torch
import warnings
warnings.filterwarnings("ignore") # TODO: remove this in final release
from utils.utils import fix_lists, fix_band_list, get_data_files_pre_train, string_to_boolean
from utils.utils import get_data_files_fine_tune # TODO: remove this, just for test
from utils.dataloader import Seviri_Dataset
from models.MaskedAutoEncoderViT.MaskedAutoEncoderViTModel import MaskedAutoEncoderViTModel
from torch.utils.data import DataLoader


def pre_train(
        dataset_path_images, 
        dataset_path_masks,
        data_extension, 
        pre_train_years, 
        pre_train_months, 
        pre_train_exclude_days,
        include_bands,
        checkpoint_path,
        learning_rate,
        batch_size,
        mask_ratio,
        loss_function,
        class_weights,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_pixel_loss):


        # load the data file names (paths)
        data_files_images = get_data_files_pre_train(dataset_path_images, data_extension, pre_train_years, pre_train_months, pre_train_exclude_days)

        # define pre-train dataset | set "mode" to "pre-train" for pre-training
        pre_train_dataset = Seviri_Dataset(filenames=data_files_images, mode='pre-train', bands=include_bands)
        #print(f"Total images for pre-train: {len(pre_train_dataset)}") # TODO: enable print
       


        # TODO: fix: UserWarning: Creating a tensor from a list of numpy.ndarrays 
        # is extremely slow. Please consider converting the list to a single 
        # numpy.ndarray with numpy.array() before converting to a tensor. 

        # define dataloader and load files into ram
        pre_train_data = DataLoader(pre_train_dataset, 
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

        # put model into train mode
        model.train()


        # using AdamW optimizer for ViT models
        #optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate) # TODO: add weight decay and betas
        #optimizer.zero_grad()


        # main pre-train loop
        for image in pre_train_data:
            # pass input data to gpu device (if cuda was available)
            image = image.to(device)

            output = model(image)










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
    checkpoint_path        = pre_train_config['pre-train']['checkpoint_save_path']
    batch_size             = pre_train_config['pre-train']['batch_size']
    learning_rate          = pre_train_config['pre-train']['learning_rate']
    mask_ratio             = pre_train_config['pre-train']['mask_ratio']
    loss_function          = pre_train_config['pre-train']['loss_function']
    class_weights          = pre_train_config['pre-train']['class_weights']

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
    include_bands = fix_band_list(include_bands)

    patch_size = int(patch_size)
    learning_rate = float(learning_rate)
    batch_size = int(batch_size)
    mask_ratio = float(mask_ratio)
    class_weights = fix_band_list(class_weights)
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
        include_bands,
        checkpoint_path,
        learning_rate,
        batch_size,
        mask_ratio,
        loss_function,
        class_weights,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_pixel_loss)  
