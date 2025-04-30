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


class Seviri_Dataset(Dataset):
    def __init__(self, filenames, mode, bands):

        self.filenames        = filenames
        self.mode             = mode
        self.bands            = bands
        self.fill_nan_value   = 0 # nan values are filled with zero
        self.data             = []
        self.load_data()



    def fill_nan_values(self, data):
        return data.fillna(self.fill_nan_value)

    def select_bands(self, data):
        return data.sel(band=self.bands)
        

    def load_data(self):
         
        # load only images 
        if self.mode == "pre-train":

            for file_image in self.filenames:
                # open image using xarray (for seviri we open .tif files)
                image_data = xr.open_dataset(file_image)

                # select specific bands
                image_data = self.select_bands(image_data)

                # fill nan values
                image_data = self.fill_nan_values(image_data)

                # put image data to a list for all bands and normalize
                image_data_array = []
                for band_idx in image_data.band-1:
                    data_array = image_data.isel(band=band_idx)['band_data'].values

                    # normalize in range of [0, 1]
                    data_array_min_value = data_array.min()
                    data_array_max_value = data_array.max()
                    data_array_normalized = (data_array - data_array_min_value) / (data_array_max_value - data_array_min_value)

                    image_data_array.append(data_array_normalized)

                self.data.append(image_data_array)
                
                # close data source
                image_data.close()    

        elif self.mode == "fine-tune":

            for files in self.filenames:
                
                # split image and mask paths
                image_data_file, mask_data_file = files[0], files[1]

                # open image using xarray (for seviri we open .tif files)
                image_data = xr.open_dataset(image_data_file)

                # select specific bands
                image_data = self.select_bands(image_data)

                # fill nan values
                image_data = self.fill_nan_values(image_data)

                # put image data to a list for all bands and normalize
                image_data_array = []
                for band_idx in image_data.band-1:
                    data_array = image_data.isel(band=band_idx)['band_data'].values

                    # normalize in range of [0, 1]
                    data_array_min_value = data_array.min()
                    data_array_max_value = data_array.max()
                    data_array_normalized = (data_array - data_array_min_value) / (data_array_max_value - data_array_min_value)

                    image_data_array.append(data_array_normalized)


                # load mask files for fine-tune
                # load mask using xarray (for seviri its .tif files)
                mask_data = xr.open_dataset(mask_data_file)

                # put mask data into a list
                mask_data_array = []
                mask_data_array.append(mask_data.isel(band=0)['band_data'].values)

                # TODO : Add normalization for masks
                
                # close data source
                image_data.close() 
                mask_data.close()

                self.data.append((image_data_array, mask_data_array))



    def __len__(self):
        return len(self.filenames_images)


    def __getitem__(self, idx):
        if self.mode == "pre-train":
            image = self.data[idx]
            return torch.tensor(image, dtype=torch.float32)
            
        elif self.mode == "fine-tune":
            image, mask = self.data[idx][0], self.data[idx][1]
            # convert input image and mask to torch tensors and return
            return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


    
     
