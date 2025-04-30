# This is a simple dataloader for the seviri data
# it makes use of xarray to open the data and load them 
# it reutrns a tesnor, is can be used for both
# images and masks
# =======================================================

import os
import xarray as xr
import torch
import numpy as np
from torch.utils.data import Dataset


class Generic_Dataset(Dataset):
    def __init__(self, filename_image, filename_mask, bands):
        self.filename_image = filename_image
        self.filename_mask  = filename_mask
        self.bands          = bands
        self.fill_nan_value = 0 # nan values are filled with zero
        self.data           = []
        self.load_data()



    def fill_nan_values(self, data):
        return data.fillna(self.fill_nan_value)

    def select_bands(self, data):
        return data.sel(band=self.bands)

    def select_dates(self, data):
        # TODO: add selection by dates for the images and masks
        return 0
        

    def load_data(self):    
        # open image using xarray (for seviri we open .tif files)
        image_data = xr.open_dataset(self.filename_image)

        # select specific bands
        image_data = self.select_bands(image_data)

        # fill nan values
        image_data = self.fill_nan_values(image_data)

        # put image data to a list for all bands
        image_data_array = []
        for band_idx in image_data.band-1:
            image_data_array.append(image_data.isel(band=band_idx)['band_data'].values)

        # load mask using xarray (for seviri its .tif files)
        mask_data = xr.open_dataset(self.filename_mask)

        # put mask data into a list
        mask_data_array = []
        mask_data_array.append(mask_data.isel(band=0)['band_data'].values)
        
        # close data sources
        image_data.close()
        mask_data.close()

        # append the image list and mask list to a list
        self.data.append((image_data_array, mask_data_array))
        
        return 0



    def __len__(self):
        return len(self.filename_image)


    def __getitem__(self, idx):
        image, mask = self.data = [idx]

        # convert input image and mask to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tesnor(mask, dtype=torch.float32)

        return image, msk


    
        

if __name__ == "__main__":
    os.system('clear')
    print('DataLoader testing...')

    images_path = "dataset/images/20230506131241_patch_203.tif"
    masks_path = "dataset/masks/20230506131241_patch_203.tif"

    bands = [1, 2, 3]

    data = Generic_Dataset(images_path, masks_path, bands)  

    #data.load_data()      
