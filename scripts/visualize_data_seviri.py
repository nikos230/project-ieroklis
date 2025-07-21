import os
import matplotlib.pyplot as plt
import xarray as xr
import yaml
import glob


if __name__ == "__main__":
    os.system("clear")

    with open('configs/dataset.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    file.close()    

    path_to_images = dataset_config['dataset']['images_path']
    path_to_masks  = dataset_config['dataset']['masks_path']
    
    data_files_images = glob.glob(path_to_images + '/*' + '.tif')
    data_files_masks = glob.glob(path_to_masks + '/*' + '.tif')

    print(f'Number of SEVIRI images: {len(data_files_images)}')
    print(f'Number of SEVIRI masks: {len(data_files_masks)}')


    # count images per years and months
    years = {}
    months = {}
    months_years = {}

    for image in data_files_images:
        # extract the date from file name
        file_name = os.path.basename(image)
        
        # extract the timedate string from the filename
        timedate = file_name.split('_')[0]

        # extract the year from the timedate string
        year = str(timedate[0] + timedate[1] + timedate[2] + timedate[3])

        # extract the month from timedate string
        month = str(timedate[4] + timedate[5])

        # extract the day fro timedate string
        day = str(timedate[6] + timedate[7])

        # track number of images per year
        if year not in years:
            years[year] = 1
        else:
            years[year] += 1    

        # track number of samples per month for all years
        if month not in months:
            months[month] = 1
        else:
            months[month] += 1        

        # track number of images per month for every year
        if year not in months_years:
            months_years[year] = {}

        if month not in months_years[year]:
            months_years[year][month] = 1
        else:
            months_years[year][month] += 1    

    print(f'Seviri Images per Year: {sorted(years.items())}\n')
    print(f'Seviri Images per Month (All Years combined): {sorted(months.items())}\n')
    print(f'Seviri Images per Month: {sorted(months_years.items())}')
    exit()

