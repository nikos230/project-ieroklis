# This is a simple dataloader for the MedST-28 dataset
# it makes use of xarray to open the data and load them 
# it returns a tensor, is can be used for both
# images and masks
# =======================================================

import os
import xarray as xr
import torch
import numpy as np
import albumentations as A
import json
import pandas as pd
from torch.utils.data import Dataset


class MedST28_Dataset(Dataset):
    def __init__(self, filenames, mode, variables, time_steps, stats_path, reference_date):

        self.filenames        = filenames
        self.stats_path       = stats_path
        self.stats            = None
        self.mode             = mode
        self.time_steps       = time_steps
        self.variables        = variables
        self.fill_nan_value   = 0 # nan values are filled with zero
        self.data             = [] 
        self.reference_date   = reference_date

        # define transformation for data augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'mask': 'mask'})
        
        # open and load normalize stats from file
        with open(stats_path, 'r') as file:
            self.stats = json.load(file)
        file.close()   
        
        # begin loading data
        self.load_data()
        
    def fill_nan_values(self, data):
        return data.fillna(self.fill_nan_value)
        
    def normalize(self, data, var_name):
        mean = self.stats[var_name]["mean"]
        std = self.stats[var_name]["std"]

        if std != 0:
            data = ( data - mean ) / std
        else:
            print(f"Found std = 0 ! for variable {var_name}")
            exit()

        return data       


    def load_data(self):
        # load only images 
        if self.mode == "pre-train":

            for sample in self.filenames:
                # open image using xarray (for seviri we open .tif files)
                sample_data = xr.open_dataset(sample)

                # fill nan values
                sample_data = self.fill_nan_values(sample_data)

                # define how many time steps will be used
                if self.time_steps == 'max':
                    self.time_steps = sample_data.dims['time']
                else:
                    if self.time_steps <= sample_data.dims['time']:
                        self.time_steps = int(self.time_steps)    
                    else:
                        print(f"Max time steps for this sample {sample_data.dims['time']}, user input was {self.time_steps}")    
                        exit()

                # put image data to a list for all bands and normalize
                data_array = []

               
                # append all static variables
                static_vars = []
                for var_name in self.variables:
                    data_static = sample_data[var_name]
                    if 'time' not in data_static.dims:
                        data_static_array = data_static.values
                        data_static_array = self.normalize(data_static_array, var_name)
                        static_vars.append(data_static_array)

                # stack all static variable 
                static_vars_stack = np.stack(static_vars, axis=0)

                # append all dynamic variables
                for time_step in range(0, self.time_steps):
                    time_steps_vars = []

                    for var_name in self.variables:
                        data_time_step = sample_data[var_name]
                        if 'time' in data_time_step.dims:
                            #print(var_name)
                            data_time_step_array = data_time_step.values[time_step]
                            data_time_step_array = self.normalize(data_time_step_array, var_name)
                            time_steps_vars.append(data_time_step_array)
   
                    # append static vars for this time steps
                    time_steps_vars.extend(static_vars_stack)

                    # create tensor
                    time_step_tensor = np.stack(time_steps_vars, axis=0)

                    # append the tensor to the data array
                    data_array.append(time_step_tensor)


                # get time info from sample and convert it to days from reference date
                initial_time_stamp = sample_data.attrs['date_start']
                initial_time_stamp = pd.to_datetime(initial_time_stamp)
                time_stamp_reference = pd.Timestamp(self.reference_date)
                time_delta = pd.Timedelta(days=1) # TODO: calculate this somehow

                time_stamps = []
                time_stamp = initial_time_stamp
                for time_step in range(0, self.time_steps):
                    time_stamp_relative = (time_stamp - time_stamp_reference).days
                    time_stamps.append(torch.tensor(time_stamp_relative, dtype=torch.float32))

                    time_stamp += time_delta
                    

                # append the sample the data list
                self.data.append([data_array, time_stamps])
                
                # close data source
                sample_data.close()    


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
            data = np.array(self.data[idx][0]) 
            time_stamps = self.data[idx][1]

            # apply transforms
            data_transformed = []

            for time_step in range(0, data.shape[0]):
                time_step_vars = []
                variables = data[time_step]
                for variable in variables:
                    #variable = np.transpose(variable, (1, 2, 0))
                    transformed = self.transform(image=variable)
                    transformed_variable = transformed["image"]
                    #transformed_variable = np.transpose(transformed_variable["image"], (2, 0, 1))
                    transformed_variable = torch.tensor(transformed_variable, dtype=torch.float32)
                    time_step_vars.append(transformed_variable)

                time_step_tensor = torch.stack(time_step_vars)
                data_transformed.append(time_step_tensor)

            return torch.stack(data_transformed), torch.stack(time_stamps)

            
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


    
     
