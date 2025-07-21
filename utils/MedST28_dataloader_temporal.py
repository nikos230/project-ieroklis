import os
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
        # min = self.stats[var_name]["min"]
        # max = self.stats[var_name]["max"]

        # if max != min:
        #     data = ( data - min ) / ( max - min )
        #     return data
        # else:
        #     return data

        mean = self.stats[var_name]["mean"]
        std = self.stats[var_name]["std"]

        return ( data - mean ) / ( std + 1e-6 )


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
                    static_vars.append(torch.tensor(arr, dtype=torch.float32))
            static_stack = np.stack(static_vars, axis=0) if static_vars else None


            for t in range(time_steps):
                time_vars = []
                for var_name in self.variables:
                    var = sample_data[var_name]
                    if 'time' in var.dims:
                        val = self.normalize(var.values[t], var_name)
                        time_vars.append(torch.tensor(val, dtype=torch.float32))
                if static_stack is not None:
                    time_vars.extend(static_stack)
                time_tensor = np.stack(time_vars, axis=0)
                data_array.append(time_tensor)


            # get day of year (doy) from sample
            dt = datetime.datetime.fromisoformat(sample_data.attrs['date_start'])
            day_of_year = dt.timetuple().tm_yday

            # Time stamp conversion
            time_stamps = []
            start = pd.to_datetime(sample_data.attrs['date_start'])
            ref = pd.Timestamp(self.reference_date)
            for t in range(time_steps):
                delta = (start + pd.Timedelta(days=t) - ref).days
                day_of_year += 1
                time_stamps.append(torch.tensor([delta, day_of_year], dtype=torch.float32))
                

            sample_data.close()

            # Apply augmentations
            # data_transformed = []
            # for step_tensor in data_array:
            #     step_vars = []
            #     for var in step_tensor:
            #         #aug = self.transform(image=var)
            #         #var_tensor = torch.tensor(aug["image"], dtype=torch.float32)
            #         var_tensor = torch.tensor(var, dtype=torch.float32)
            #         step_vars.append(var_tensor)
            #     data_transformed.append(torch.stack(step_vars))

            return torch.tensor(data_array, dtype=torch.float32), torch.stack(time_stamps)


        elif self.mode == "fine-tune-fire_spread":
            sample_path = self.filenames[idx]
            sample_data = xr.open_dataset(sample_path)
            sample_data = sample_data.isel(time=slice(4, None))
            sample_data = self.fill_nan_values(sample_data)
            
            max_time = sample_data.dims['time']
            time_steps = max_time if self.time_steps == 'max' else min(self.time_steps)
      
            data_array = []

            # zero all igntions points from all days execpt day 4
            # day4_igntion_point = sample_data['ignition_points'].values[4, :, :].copy()
            # sample_data['ignition_points'][:] = 0
            # sample_data['ignition_points'].values[4, :, :] = day4_igntion_point


            # Static variables
            static_vars = []
            for var_name in self.variables:
                var = sample_data[var_name]
                if 'time' not in var.dims:
                    arr = self.normalize(var.values, var_name)
                    static_vars.append(torch.tensor(arr, dtype=torch.float32))
            static_stack = np.stack(static_vars, axis=0) if static_vars else None

            for t in range(0, max_time):
                time_vars = []
                for var_name in self.variables:
                    var = sample_data[var_name]
                    if 'time' in var.dims:
                        val = self.normalize(var.values[t], var_name)
                        time_vars.append(torch.tensor(val, dtype=torch.float32))
                if static_stack is not None:
                    time_vars.extend(static_stack)
                time_tensor = np.stack(time_vars, axis=0)
                data_array.append(time_tensor)

            
             # get day of year (doy) from sample
            dt = datetime.datetime.fromisoformat(sample_data.attrs['date'])
            day_of_year = dt.timetuple().tm_yday

            # Time stamp conversion
            time_stamps = []
            start = pd.to_datetime(sample_data.attrs['date'])
            ref = pd.Timestamp(self.reference_date)
            for t in range(time_steps):
                delta = (start + pd.Timedelta(days=t) - ref).days
                day_of_year += 1
                time_stamps.append(torch.tensor([delta, day_of_year], dtype=torch.float32))


            # label variable
            label = sample_data[self.label].values[0, :, :] # get label for time=4 (first fire day) 
            label = (label > 0).astype(np.float32)

            
            sample_data.close()

            # Apply augmentations
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
            

            #return torch.stack(data_transformed), torch.stack(time_stamps), torch.tensor(label, dtype=torch.float32)
            return torch.tensor(data_array, dtype=torch.float32), torch.stack(time_stamps), torch.tensor(label, dtype=torch.float32)
        


        elif self.mode == "fine-tune-fire_risk":
            sample_path = self.filenames[idx]
            sample_data = xr.open_dataset(sample_path)
            sample_data = self.fill_nan_values(sample_data)

            max_time = sample_data.dims['time']
            time_steps = max_time if self.time_steps == 'max' else min(self.time_steps, max_time)

            data_array = []

            # zero all igntions points from all days execpt day 4
            day4_igntion_point = sample_data['ignition_points'].isel(time=4)
            sample_data['ignition_points'][:] = 0
            sample_data['ignition_points'].values[4, :, :] = day4_igntion_point

            # Static variables
            static_vars = []
            for var_name in self.variables:
                var = sample_data[var_name]
                if 'time' not in var.dims:
                    arr = self.normalize(var.values, var_name)
                    static_vars.append(arr)
            static_stack = np.stack(static_vars, axis=0) if static_vars else None

            for t in range(time_steps):
                time_vars = []
                for var_name in self.variables:
                    var = sample_data[var_name]
                    if 'time' in var.dims:
                        val = self.normalize(var.values[t], var_name)
                        time_vars.append(val)
                if static_stack is not None:
                    time_vars.extend(static_stack)
                time_tensor = np.stack(time_vars, axis=0)
                data_array.append(time_tensor)


            # Time stamp conversion
            time_stamps = []
            start = pd.to_datetime(sample_data['time'].values[0])
            ref = pd.Timestamp(self.reference_date)
            for t in range(time_steps):
                delta = (start + pd.Timedelta(days=t) - ref).days
                time_stamps.append(torch.tensor(delta, dtype=torch.float32))


            # label variable
            if sample_data['burned_area_has'].values.max() > 0:
                label = 1
            else:
                label = 0    
            label = torch.tensor(label, dtype=torch.float32)

            sample_data.close()

            # Apply augmentations
            data_transformed = []
            for step_tensor in data_array:
                step_vars = []
                for var in step_tensor:
                    #aug = self.transform(image=var, label=label)
                    #var_tensor = torch.tensor(aug["image"], dtype=torch.float32)
                    #label = torch.tensor(aug["label"])
                    var_tensor = torch.tensor(var, dtype=torch.float32)
                    step_vars.append(var_tensor)
                data_transformed.append(torch.stack(step_vars))
            


            return torch.stack(data_transformed), torch.stack(time_stamps), label
