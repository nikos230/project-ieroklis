import os
from random import sample
import xarray as xr
import torch
import numpy as np
import albumentations as A
import json
import pandas as pd
import datetime
from torch.utils.data import Dataset


class MedST28_Dataset(Dataset):
    def __init__(self, filenames, mode, training_data, variables, time_steps, stats_path, reference_date, label=None, samples_patch_size='max'):
        self.filenames = filenames
        self.mode = mode
        self.variables = variables
        self.time_steps = time_steps
        self.samples_patch_size = samples_patch_size
        self.fill_nan_value = 0
        self.reference_date = reference_date
        self.stats_path = stats_path
        self.label = label
        self.training_data = training_data

        # Load normalization stats
        with open(stats_path, 'r') as file:
            self.stats = json.load(file)

        # Define augmentations
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'mask': 'mask', 'label': 'label'})



    def fill_nan_values(self, data):
        return data.fillna(self.fill_nan_value)

    def normalize(self, data, var_name):
        # mean = self.stats[var_name]["mean"]
        # std = self.stats[var_name]["std"]
        
        # return ( data - mean ) / ( std + 1e-6 )

        min = self.stats[var_name]["min"]
        max = self.stats[var_name]["max"]

        if min != max:
            return ( data - min ) / (max - min)
        else:
            return data 


        


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.mode == "pre-train":
            sample_path = self.filenames[idx]
            sample_data = xr.open_dataset(sample_path)

            sample_data = self.fill_nan_values(sample_data)

            # select a smaller patch of the sample if defined from config
            if self.samples_patch_size != 'max':
                x_center = sample_data.dims['x'] // 2
                y_center = sample_data.dims['y'] // 2

                x_start = x_center - 32
                x_end = x_center + 32
                y_start = y_center - 32
                y_end = y_center + 32

                sample_data = sample_data.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
            

            max_time = sample_data.dims['time']
            time_steps = max_time if self.time_steps == 'max' else int(self.time_steps)

            data_array = []

            # Static variables
            static_vars = []
            for var_name in self.variables:
                var = sample_data[var_name]
                if 'time' not in var.dims:
                    arr = self.normalize(var.values, var_name)
                    data_array.append(torch.tensor(arr, dtype=torch.float32))
                    

            for t in range(time_steps):
                for var_name in self.variables:
                    var = sample_data[var_name]
                    if 'time' in var.dims:
                        val = self.normalize(var.values[t], var_name)
                        data_array.append(torch.tensor(val, dtype=torch.float32))

            sample_data.close()


            # Apply augmentations
            if self.training_data == True:
                # Stack inputs: (C, H, W) → (H, W, C)
                input_np = np.stack([arr.numpy() for arr in data_array], axis=0)
                input_np = np.transpose(input_np, (1, 2, 0))
                augmented = self.transform(image=input_np)

                input_aug = np.transpose(augmented["image"], (2, 0, 1))  # (C, H, W)

                
                return torch.tensor(input_aug, dtype=torch.float32)
            
            else:
               return torch.stack(data_array)

 



        elif self.mode == "fine-tune-fire_spread":
            sample_path = self.filenames[idx]
            sample_data = xr.open_dataset(sample_path)
            #sample_data = sample_data.isel(time=slice(4, None))
            sample_data = self.fill_nan_values(sample_data)
            
            max_time = sample_data.dims['time']

            #time_steps = max_time if self.time_steps == 'max' else min(self.time_steps, max_time)

            data_array = []

            # zero all igntions points from all days execpt day 4
            day4_igntion_point = sample_data['ignition_points'].values[4, :, :].copy()
            sample_data['ignition_points'][:] = 0
            sample_data['ignition_points'].values[4, :, :] = day4_igntion_point

            # Static variables
            static_vars = []
            for var_name in self.variables:
                var = sample_data[var_name]
                if 'time' not in var.dims:
                    arr = self.normalize(var.values, var_name)
                    data_array.append(torch.tensor(arr, dtype=torch.float32))
            
            for t in range(0, max_time):
                time_vars = []
                for var_name in self.variables:
                    var = sample_data[var_name]
                    if 'time' in var.dims:
                        if var_name == 'ignition_points':
                            data_array.append(torch.tensor(var.values[t], dtype=torch.float32))
                            continue    
                        val = self.normalize(var.values[t], var_name)
                        data_array.append(torch.tensor(val, dtype=torch.float32))
           
            # label variable
            label = sample_data[self.label].values[4, :, :] # get label for time=4 (first fire day) 
            label = (label > 0).astype(np.float32)

            
            sample_data.close()

            # Apply augmentations
            if self.training_data == True:
                # Stack inputs: (C, H, W) → (H, W, C)
                input_np = np.stack([arr.numpy() for arr in data_array], axis=0)
                input_np = np.transpose(input_np, (1, 2, 0))
                augmented = self.transform(image=input_np, mask=label)

                input_aug = np.transpose(augmented["image"], (2, 0, 1))  # (C, H, W)
                label_aug = augmented["mask"]
                
                return torch.tensor(input_aug, dtype=torch.float32), torch.tensor(label_aug, dtype=torch.float32)
            
            else:
               return torch.stack(data_array), torch.tensor(label, dtype=torch.float32)
        


        elif self.mode == "fine-tune-fire_risk":
            sample_path = self.filenames[idx]
            sample_data = xr.open_dataset(sample_path)
            sample_data = sample_data.isel(time=slice(20, None))
            sample_data = self.fill_nan_values(sample_data)
       
            max_time = sample_data.dims['time']
            #time_steps = max_time if self.time_steps == 'max' else min(self.time_steps, max_time)

            # define static and dynamic vars (remove later)    
            static_var_names = ['aspect', 'curvature', 'dem', 
                                'roads_distance', 'slope', 'lc_agriculture', 
                                'lc_forest', 'lc_grassland', 'lc_settlement', 
                                'lc_shrubland', 'lc_sparse_vegetation', 'lc_water_bodies', 
                                'lc_wetland', 'population' ]
            
            dynamic_var_names = ['d2m', 'ignition_points', 'lai', 
                                 'lst_day', 'lst_night', 'ndvi', 
                                 'rh', 'smi', 'sp', 
                                 'ssrd', 't2m', 'tp', 
                                 'wind_direction', 'wind_speed']

            data_array = []

            # Static variables
            static_vars = []
            for var_name in static_var_names: # self.variables
                var = sample_data[var_name]
                #if 'time' not in var.dims:
                arr = self.normalize(var.values[0], var_name)
            
                #static_vars.append(torch.tensor(arr, dtype=torch.float32))
                data_array.append(torch.tensor(arr, dtype=torch.float32))
            #static_stack = np.stack(static_vars, axis=0) if static_vars else None

            for t in range(0, max_time):
                time_vars = []
                for var_name in dynamic_var_names: # self.variables
                    var = sample_data[var_name]
                    if 'time' in var.dims:
                        val = self.normalize(var.values[t], var_name)
                        #time_vars.append(torch.tensor(val, dtype=torch.float32))
                        data_array.append(torch.tensor(val, dtype=torch.float32))

                # if static_stack is not None:
                #     time_vars.extend(static_stack)
                #time_tensor = np.stack(time_vars, axis=0)
                #data_array.append(time_tensor)

            # data_array = []

            # # Static variables
            # static_vars = []
            # for var_name in static_var_names:
            #     var = sample_data[var_name]
            #     arr = self.normalize(var.values[0], var_name)
            #     static_vars.append(torch.tensor(arr, dtype=torch.float32))
            # static_stack = torch.stack(static_vars, dim=0).flatten()  # shape: [static_feat_dim]

            # time_series_data = []

            # for t in range(0, max_time):
            #     time_vars = []
            #     for var_name in dynamic_var_names:
            #         var = sample_data[var_name]
            #         if 'time' in var.dims:
            #             val = self.normalize(var.values[t], var_name)
            #             time_vars.append(torch.tensor(val, dtype=torch.float32))

            #     dynamic_stack = torch.stack(time_vars, dim=0).flatten()  # [dynamic_feat_dim]
            #     full_features = torch.cat([dynamic_stack, static_stack], dim=0)  # [full_feat_dim]
            #     time_series_data.append(full_features)

            # time_series_tensor = torch.stack(time_series_data, dim=0) 
    

            # label variable
            if sample_data['burned_area_has'].values.max() > 0:
                class_weights = torch.tensor(np.log1p(sample_data['burned_area_has'].values.max()), dtype=torch.float32)
                label = 1

            else:
                class_weights = torch.tensor(1, dtype=torch.float32)
                label = 0   
            label = torch.tensor(label, dtype=torch.long)

            sample_data.close()
           
            # # Apply augmentations
            # data_transformed = []
            # for step_tensor in data_array:
            #     step_vars = []
            #     for var in step_tensor:
            #         #aug = self.transform(image=var, label=label)
            #         #var_tensor = torch.tensor(aug["image"], dtype=torch.float32)
            #         #label = torch.tensor(aug["label"])
            #         var_tensor = torch.tensor(var, dtype=torch.float32)
            #         step_vars.append(var_tensor)
            #     data_transformed.append(torch.stack(step_vars))
            

            # return time_series_tensor, label, class_weights
            return torch.stack(data_array), label, class_weights
