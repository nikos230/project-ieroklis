from cgi import test
import os
from numpy.ma import count
from timm.data import dataset
import xarray as xr
import json
import yaml
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm


if __name__ == "__main__":
    os.system('clear')
    
    # with open('configs/MedST-28_configs/MedST28_dataset.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    # file.close() 
    
    # with open('configs/MedST-28_configs/WildfireSpread_dataset.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    # file.close() 

    with open('configs/MedST-28_configs/FireRisk_dataset.yaml', 'r') as file:
        config = yaml.safe_load(file)
    file.close() 


    # path to dataset folder
    #dataset_path = config['dataset']['path_to_MedST28_dataset']
    dataset_path = config['dataset']['path_to_dataset']
    path_to_stats = config['dataset']['path_to_stats']

    # get validation and test years to exclude them later from the calculation of the stats
    validation_years = config['dataset']['validation']['include_years']
    validation_years = [validation_years][0].split(', ')

    test_years = config['dataset']['test']['include_years']
    test_years = [test_years][0].split(', ')

    # add all excluded years
    excluded_folders = validation_years + test_years

    # find all .nc files
    file_names = [
    f for f in glob.glob(f"{dataset_path}/**/*.nc", recursive=True)
    if not any(excluded in f.split(os.sep) for excluded in excluded_folders)
    ]
    print(f"Total Number of Samples: {len(file_names)}")
    
    stats = defaultdict(lambda: {"count": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")})

    for file in tqdm(file_names):
        ds = xr.open_dataset(file)

        # fill nan with 0
        #ds = ds.fillna(0)

        for var in ds.data_vars:
            data = ds[var].values
            flat = data.ravel()
            flat = flat[~np.isnan(flat)]
            
            n = len(flat)
            if n == 0:
                continue

            mean_old = stats[var]["mean"]
            count_old = stats[var]["count"]
            M2_old = stats[var]["M2"]

            count_new = count_old + n

            delta = flat.mean() - mean_old
            mean_new = mean_old + delta * n / ( count_new )

            M2_new = M2_old + np.sum((flat - mean_old) * (flat - mean_new))


            # update stats
            stats[var]["count"] = count_new
            stats[var]["mean"] = mean_new
            stats[var]["M2"] = M2_new
            stats[var]["min"] = min(stats[var]["min"], flat.min())
            stats[var]["max"] = max(stats[var]["max"], flat.max())

        ds.close()    


    # final stats
    final_stats = {}
    for var, s in stats.items():
        mean = s["mean"]
        std = np.sqrt(s["M2"] / s["count"])
        min_val = s["min"]
        max_val = s["max"]
        final_stats[var] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min_val),
            "max": float(max_val)
        }

    with open(f"{path_to_stats}", 'w') as file:
        json.dump(final_stats, file, indent=2)