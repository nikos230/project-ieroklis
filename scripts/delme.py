import os
import xarray as xr
import matplotlib.pyplot as plt



if __name__ == "__main__":
    os.system('clear')

    path_to_image = 'dataset/seviri_dataset_testing/images/20230506131241_patch_203.tif'

    ds = xr.open_dataset(path_to_image)
    #ds = ds.sel(band=[1, 2, 3, 4, 8, 9])

    # fig, axis = plt.subplots(2, 7)
    
    # axis[0, 0].imshow(ds['band_data'][0].values)
    # axis[0, 1].imshow(ds['band_data'][1].values)
    # axis[0, 2].imshow(ds['band_data'][2].values)
    # axis[0, 3].imshow(ds['band_data'][3].values)
    # axis[0, 4].imshow(ds['band_data'][4].values)
    # axis[0, 5].imshow(ds['band_data'][5].values)
    # axis[0, 6].imshow(ds['band_data'][6].values)
    # axis[1, 0].imshow(ds['band_data'][7].values)
    # axis[1, 1].imshow(ds['band_data'][8].values)
    # axis[1, 2].imshow(ds['band_data'][9].values)
    # axis[1, 3].imshow(ds['band_data'][10].values)
    # axis[1, 4].imshow(ds['band_data'][11].values)
    # axis[1, 5].imshow(ds['band_data'][12].values)
    # axis[1, 6].imshow(ds['band_data'][13].values)

    fig, axis = plt.subplots(2, 7)
    axis = axis.flatten()
    for i in range(0, 14):

        axis[i].imshow(ds['band_data'][i].values)
        

    plt.savefig('output/test2.png')

    print(ds)