import os
from random import sample
from traceback import print_tb
from xml.sax.expatreader import ExpatLocator
import xarray as xr
import pandas as pd
import numpy as np


if __name__ == "__main__":
    os.system('clear')

    path_to_new_dataset = 'dataset/FireRisk_dataset_netCDF_new'
    os.makedirs(path_to_new_dataset, exist_ok=True)

    #path_to_positives = 'dataset/FireRisk_dataset/positives.csv'
    path_to_negatives = 'dataset/FireRisk_dataset/negatives.csv'

    df = pd.read_csv(path_to_negatives)

    df['time'] = pd.to_datetime(df['time'])

    dims = ['time', 'y', 'x']
    
    excluded_cols = {'time', 'x', 'y', 'time_idx', 'sample'}

    data_vars = [col for col in df.columns if col not in excluded_cols]

    sample_ids = df['sample'].unique()

    counter = 0
    for sample_id in sample_ids:
        df_sample = df[df['sample'] == sample_id].copy()

        time_vals = np.sort(df_sample['time'].unique())
        x_vals = np.sort(df_sample['x'].unique())
        y_vals = np.sort(df_sample['y'].unique())

        time_index = {v: i for i, v in enumerate(time_vals)}
        x_index = {v: i for i, v in enumerate(x_vals)}
        y_index = {v: i for i, v in enumerate(y_vals)}
        
        year = str(time_vals[0].astype('datetime64[Y]'))

        shape = (len(time_vals), len(x_vals), len(y_vals))

        data_arrays = {var: np.full(shape, np.nan, dtype='float32') for var in data_vars}
        
        for row in df_sample.itertuples(index=False):
            t = time_index[row.time]
            x = x_index[row.x]
            y = y_index[row.y]
            
            for var in data_vars:
                data_arrays[var][t, x, y] = getattr(row, var)

        ds = xr.Dataset(
            {var: (['time', 'x', 'y'], data_arrays[var]) for var in data_vars},
            coords={
                'time': time_vals,
                'x': x_vals,
                'y': y_vals}    
            )        

        file_name = f"sample_neg_{sample_id}.nc"
        save_path = os.path.join(path_to_new_dataset, year)
        save_path_file = os.path.join(save_path, file_name)
        os.makedirs(save_path, exist_ok=True)

        ds.to_netcdf(save_path_file, engine='h5netcdf')

        counter += 1
        
    print('Done!')