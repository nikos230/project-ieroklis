# This is a simple dataloader for the seviri data
# it makes use of xarray to open the data and load them 
# it returns a tensor, is can be used for both
# images and masks
# =======================================================

import os
import xarray as xr
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Seviri_Dataset(Dataset):
    def __init__(self, filenames, mode, bands):

        self.filenames        = filenames
        self.mode             = mode
        self.bands            = bands
        self.fill_nan_value   = 0 # nan values are filled with zero
        self.data             = []
        self.load_data()

        # define transformation for data augmetation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'mask': 'mask'})


    def fill_nan_values(self, data):
        return data.fillna(self.fill_nan_value)

    def select_bands(self, data):
        #print(self.bands)
        #print(data.sel(band=self.bands))
        #print(data.sel(band=self.bands))
        #exit(22)
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
                
                for band_idx in range(0, len(self.bands)):
                    data_array = image_data['band_data'][band_idx].values

                    # normalize in range of [0, 1]
                    data_array_min_value = data_array.min()
                    data_array_max_value = data_array.max()

                    if data_array_max_value > data_array_min_value:
                        data_array_normalized = (data_array - data_array_min_value) / (data_array_max_value - data_array_min_value)
                    else:
                        data_array_normalized = np.zeros_like(data_array)

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
                for band_idx in range(0, len(self.bands)):
                    data_array = image_data['band_data'][band_idx].values

                    # normalize in range of [0, 1]
                    data_array_min_value = data_array.min()
                    data_array_max_value = data_array.max()

                    if data_array_max_value > data_array_min_value:
                        data_array_normalized = (data_array - data_array_min_value) / (data_array_max_value - data_array_min_value)
                    else:
                        data_array_normalized = np.zeros_like(data_array)

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
        return len(self.filenames)


    def __getitem__(self, idx):
        if self.mode == "pre-train":
            image = np.array(self.data[idx]) 

            # apply transforms
            image = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image)
            image = np.transpose(augmented["image"], (2, 0, 1))

            return torch.tensor(image, dtype=torch.float32)
            
        elif self.mode == "fine-tune":
            image = np.array(self.data[idx][0])         
            mask  = np.array(self.data[idx][1][0])

            # apply transforms
            image = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image, mask=mask)
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = augmented["mask"]

            # convert input image and mask to torch tensors and return
            return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


    
     
