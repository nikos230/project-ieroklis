import os
import xarray as xr
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    os.system('clear')
    


    ds = xr.open_dataset('dataset/FireRisk_dataset_netCDF/2006/sample_0.nc')

    print(ds['lst_day'].values)



