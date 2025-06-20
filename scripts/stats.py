import os
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import numpy as np




if __name__ == "__main__":
    os.system('clear')

    with open('configs/dataset.yaml', 'r') as file:
        dataset_config = yaml.safe_load(file)
    file.close()    

    images_path = dataset_config['dataset']['images_path']
    masks_path  = dataset_config['dataset']['masks_path']

    mask_list = []
    count_images = 0
    band_values = [[] for _ in range(11)]
    band_values_zero = [[] for _ in range(11)]

    for mask_file_name in os.listdir(masks_path):
        image_file_name = mask_file_name

        if not os.path.exists(os.path.join(images_path, image_file_name)):
            print(f"Image not found for mask: {mask_file_name} — skipping.")
            continue

        count_images += 1
        
        image_ds = xr.open_dataset(os.path.join(images_path, image_file_name))
        mask_ds = xr.open_dataset(os.path.join(masks_path, mask_file_name))

        mask_data = mask_ds.isel(band=0)['band_data'].values        
        #mask_list.append(mask_data)
        
        for band_index in range(0, 11):

            band_data = image_ds.isel(band=band_index)['band_data'].values
            values = band_data[mask_data.astype(bool)]
            values_zero = band_data[~mask_data.astype(bool)]

            band_values[band_index].extend(values)

            band_values_zero[band_index].extend(values_zero)
       

    #masks_stack = np.stack(mask_list)
    
    #masks_density_map = np.mean(masks_stack, axis=0)

    # === Plot KDEs for each band ===
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))
    axes = axes.flatten()

    for b in range(11):
        sns.kdeplot(band_values[b], ax=axes[b], fill=True, bw_adjust=0.5, color='red')
        axes[b].set_title(f'Band {b+1}')
        axes[b].set_xlabel('Pixel Value')
        axes[b].set_ylabel('Density')

    for b in range(11):
        sns.kdeplot(band_values_zero[b], ax=axes[b], fill=True, bw_adjust=0.5, color='blue')
        axes[b].set_title(f'Band {b+1}')
        axes[b].set_xlabel('Pixel Value')
        axes[b].set_ylabel('Density')

    # Hide unused subplot if num_bands < len(axes)
    for ax in axes[11:]:
        ax.axis('off')

    plt.suptitle('Probability Density of Foreground Pixel Values per Band', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/stats_test.png', dpi=600)
    plt.close()


    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(masks_density_map, cmap='hot', cbar_kws={'label': 'Probability'}, xticklabels=False, yticklabels=False)
    # plt.title('Probability Density of Foreground Pixels')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.tight_layout()
    # plt.savefig('output/stats_test.png')
    # plt.close()
