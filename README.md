# Project Ieroklis 
"Project Ieroklis" has combined two wildfire downstream tasks, Fire Risk and Fire Spread, by creating a small foundation model called MedST-28 which stands for Mediterranean Spatio-Temporal 28.

## Table of Contents
- [About the Project](#about)
  - [Datasets](#Datasets)
  - [Deep Learning Models](#deep-learning-models)
  - [Models Evaluation and Metrics](#models-evaluation-and-metrics-latest-results-feb-2025)
  - [Download Best Models and Dataset](#download-best-models-and-dataset)
  - [Download Datasets for fine-tuning](#Download-links)
- [Getting Started](#getting-started)
    - [Pre-training MedST-28](#pre-training)
      - [Train UNet2D](#train-unet2d)
      - [Train UNet3D](#train-unet3d)
      - [Train UNet2D Baseline](#train-unet2d-baseline)
  - [Tesing the pre-trained Models](#tesing-the-pre-trained-models)
 - [Contributing / Contact](#contact)

## About
This repo contains ready-to-use python code and datasets to:
- **Pre-train** Masked Auto Encoders (MAE) with Vision Transformer (ViT) backbone
- **Fine-tune** pre-trained models with MAE encoder and Convolutional or LSTM decoders
- **Fire Risk** and **Fire Spread** datasets to be used in the fine-tuning of the MedST-28 model


## Getting Started
First create a new conda enviroment, with any name, here name was choosen as **medst28_env**
```
conda create -n medst28_env python=3.10
```
After installation activate the new enviroment, replace "medst28_env" with your enviroment name
```
conda activate medst28_env
```
Finally install requirements.txt
```
pip install -r requirements.txt
```
Install PyTorch separately, to avoid any errors
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Deep Learning Models
Using an Masked Auto Encoder with a Vision Transformer backbone the MedST-28 was pre-trained on a large-scale spatio-temporal dataset, consisting of 28 variables across all Mediterranean region thou years 2006 to 2019. 
From this model, two additional task-specific models were derived by retaining only the encoder part of MedST-28: an LSTM decoder was added for the Fire Risk task, and a Convolutional decoder was added for the Fire Spread task.

### MedST-28 model
The architecture used for the pre-training is shown below, its based on [SatMAE-VIT](https://github.com/sustainlab-group/SatMAE) model. To pre-train a `ViT encoder` and `decoder` are used with 75% masking in the input images. The data used for pre-training spans the years 2006 to 2019, while the years 2020, 2021, and 2022 were reserved for validation and testing in the fine-tuning tasks. To prevent data leakage, pre-training was limited to data up to 2019. <br />

`Input of pre-training`: `28 variables` in 64 x 64 pixel patches with 154 channels. <br /> <br />
The inputs consiists of 28 variables across a 10 day temporal window, so the input tensor has shape of (channels, height, width) = (154, 64, 64)


### MedST-28 fine-tune for Fire Risk
To fine-tune the MedST-28 model for the Fire Risk task, the ViT decoder was replaced with an `LSTM decoder`. This modification enables the model to learn temporal features by utilizng the features from the pre-trained encoder, the final layer of the model (after the LSTM layer) is a binary classification layer where `0 are un-burned pixels` while `1 are burned pixels`. Where were used 19553 samples for fine-tune <br />

`Input of fine-tune for Fire Risk`: 28 variables in 1 x 1 pixel patches with 154 channels* <br />
`Output of fine-tune for Fire Risk`: Binary Classification <br /> <br />

*The inputs consiists of 28 variables across a 10 day temporal window, so the input tensor has shape of (channels, height, width) = (154, 1, 1)


### MedST-28 fine-tune for Fire Spread
To fine-tune the MedST-28 model for the Fire Spread task, the ViT decoder was replaced with an `Convolutional decoder`. This modification enables the model to learn spatial features from the pre-trained encoder. The final layer of the model generates an binary segmatation map, where `0 are un-burned pixels` and `1 are burn pixels`. Where were used 7623 samples for fine-tune <br />

`Input of fine-tune Fire Spread`: 28 variables in 64 x 64 pixel patches with 154 channels* <br />
`Output of fine-tune for Fire Spread`: Binary Classification Map (Output size same as Input size, 64x64 pixels) <br /> <br />

*The inputs consiists of 28 variables across a 10 day temporal window, so the input tensor has shape of (channels, height, width) = (154, 64, 64)


| MedST-28 | MedST-28 fine-tune for Fire Risk | MedST-28 fine-tune for Fire Spread |
|----------|----------------------------------|-----------------------------------|
|          |                                  |                                   |
| <img src="https://github.com/nikos230/project-ieroklis/blob/main/misc/pre-training-setup.png" width="350"> | <img src="https://github.com/nikos230/project-ieroklis/blob/main/misc/fine_tune_fire_risk_setup.png" width="350"> | <img src="https://github.com/nikos230/project-ieroklis/blob/main/misc/fine_tune_fire_spread_setup.png" width="350"> |

## Datasets
This repo contains two datasets for fine-tuning, but not the dataset used for the pre-training of the MedST-28 model, to provide more info please contact via email (nikolas619065@gmail.com). To see which vatriables the datasets consists of and the spatial and temporal resolution please visit section [Dataset Variables, Spatial and Temporal resolutions](#dataset-variables-spatial-and-temporal-resolutions)
              
### MedST-28 pre-train dataset
For pre-training the MedST-28 model, samples were used in the form of 64 × 64 pixel (equivalent to 64 × 64 km) patches with 154 channels. Each channel corresponds to a variable at a specific time step. The 28 variables are repeated for each of the 10 days, while static variables are included only once in the patch. A total of 147,600 samples were used for pre-training and 21,600 for validation and result visualization. All models were pre-trained for 170 epochs.

### Fire Risk fine-tune dataset
To fine-tune the MedST-28 model for the Fire Risk task, samples where used in form of 1 x 1 km patches. This dataset was picked from the Mesogeos datacube [Link to paper](https://arxiv.org/pdf/2306.05144), [Link to Github repo](https://github.com/orion-ai-lab/mesogeos). Each sample consists of 28 variables and a label indicating whether the pixel is burned (value = 1) or unburned (value = 0). Each sample originally contains 10 time steps, covering the time of ignition and 10 steps prior. Out of 10 time steps time step = 10 is the ingtion day of the fire, thus each sample consists of values of every variable for 10 days before the fire starts. Variable `burned_areas_has` is used as the label in the fine-tuning and has values 0 or 1

### Fire Spread fine-tune dataset
To fine-tune the MedST-28 model for the Fire Spread task, samples were used in form of 64 x 64 pixel patchs (=64 x 64 km). This dataset was picked from WildfireSpread forecasting with Deep Learning [Link to paper](https://arxiv.org/pdf/2505.17556), [Link to Github repo](https://github.com/nikos230/WildFireSpread). Each sample consits of 27 variables and label .Out of 10 time steps time step = 4 is the ingtion day of the fire, thus each sample consists of values of every variable for 4 days before the fire starts and 5 days after the fire. Variable `burned_areas` is used as the label in the fine-tuning and represents with a binary map the final burned area

### Download links
| Dataset     | Link <br />(HuggingFace)                                                                                   | Size <br /> Zipped | Size <br /> Unzipped  |
|:-----------:|:----------------------------------------------------------------------------------------------------------:|:------------------:|:---------------------:|
| Fire Risk   | [Download](https://huggingface.co/datasets/nikos230/FireRisk/resolve/main/fire_risk_dataset_netcdf.zip)    | 148mb              | 812mb                 |
| Fire Spread | [Download](https://huggingface.co/datasets/nikos230/WildfireSpread/resolve/main/dataset_64_64_10days.zip)  | 12.1gb             | 31gb                  |


### Configure dataset paths
Download and put datasets (without changing their names) in the dataset folder, then all scripts for pre-training and fine-tune will run. If you put the dataset in another location you will have to configure their paths into the configs. <br />
- For the pre-training dataset open `MedST-28_configs/configs/MedST28_dataset.yaml` and put the new path in line 4 and dataset stats file in line 5
- For Fire Risk dataset open `MedST-28_configs/configs/FireRisk_dataset.yaml` and put the new path in line 4 and dataset stats file in line 5
- For Fire Spread dataset open `MedST-28_configs//WildfireSpread_dataset.yaml` and put the new path in line 4 and dataset stats file in line 5

For both fine-tuning datasets the stats are already calculated, which are used for normalization in the dataloader, if needed to be calculated again run `calc_norm_values_MedST28.py` and in the first lines comment or uncomment lines 17 to 27 accordingly. 

## Pre-training
To pre-train the MedST-28 model you need to download the pre-training dataset which has a size of ~500gb.
- Configure the checkpoint and visual results paths in `MedST-28_configs/configs/pre-train_config.yaml` and chnage the hyper parameters if needed (like depth, patch size, mlp_ratio, etc..)
- Configure the MedST-28 dataset settings (time_steps, patch_size, variables to be included, train years, validation years, tests years, etc..) in the `MedST-28_configs/configs/MedST28_dataset.yaml`
- Finally run `pre-train_MedST28.py`

Download pre-trained checkpoints of the model here:
| Model Size*    | Link <br />(Hugging Face)                                                           |
|:--------------:|:-----------------------------------------------------------------------------------:|
| MedST28 3.1M   | [Download](https://huggingface.co/nikos230/MedST-28/resolve/main/MedST28_3.1M.pt)   |
| MedST28 6.3M   | [Download](https://huggingface.co/nikos230/MedST-28/resolve/main/MedST28_6.3M.pt)   |
| MedST28 50M    | [Download](https://huggingface.co/nikos230/MedST-28/resolve/main/MedST28_50M.pt)    |

*3.1M, 6.3M and 50M refer to millions of parameters

## Fine-tune
The MedST-28 model can be fine-tuned for Fire Risk and Fire Spread. To add a fine-tuning task please contact via email (nikolas619065@gmail.com)

### Fire Risk fine-tune
To fine-tune the MedST-28 model for fire risk you need a pre-train checkpoint of the MedST-28 model. Choose any of the checkpoints above, best results can be achived with MedST28 3.1M.
- Configure checkpoints and pre-trained model paths in `MedST-28_configs/configs/fine-tune_config_fire_risk.yaml` in lines 5 and 6, you can also change epochs, loss function, patch_size and batch_size
- [Download](https://huggingface.co/datasets/nikos230/FireRisk/resolve/main/fire_risk_dataset_netcdf.zip) the Fire Risk dataset and configure paths as explained in ["Configure dataset paths"](Configure-dataset-paths)
- Finally run `fine-tune_MedST28_fire_risk.py`


### Fire Spread fine-tune
To fine-tune the MedST-28 model for fire spread you need a pre-train checkpoint of the MedST-28 model. Choose any of the checkpoints above, best results can be achived with MedST28 50M.
- Configure checkpoints and pre-trained model paths in `MedST-28_configs/configs/fine-tune_config_fire_spread.yaml` in lines 5 and 6, you can also change epochs, loss function, patch_size and batch_size
- [Download](https://huggingface.co/datasets/nikos230/WildfireSpread/resolve/main/dataset_64_64_10days.zip) the Fire Spread dataset and configure paths as explained in ["Configure dataset paths"](Configure-dataset-paths)
- Finally run `fine-tune_MedST28_fire_spread.py`

## Results
Results from the test tests. For Fire Risk the test samples were in total 4084 with a ratio of `Positives / Negatives = 0.5` and for Fire Spread there were 1102 samples with a ratio of `Positives / Negatives = 0.007`

| Metric (%) | MedST-28 <br /> Fire Risk 3.1M | MedST-28 <br /> Fire Risk 6.3M | MedST-28 <br /> Fire Risk 50M | MedST-28 <br /> Fire Spread 3.1M | MedST-28 <br /> Fire Spread 6.3M | MedST-28 <br /> Fire Spread 50M |
|------------|:------------------------------:|:------------------------------:|:-----------------------------:|:--------------------------------:|:--------------------------------:|:-------------------------------:|
| Accuracy   | 74.2                           | **76.6**                       | 73.4                          | **59.5**                         | 56.9                             | 55.8
| F1 Score   | **71.0**                       | 70.9                           | 68.0                          | 54.1                             | **54.6**                         | 55.3
| IoU        | **55.1**                       | 54.9                           | 51.6                          | 37.1                             | 37.5                             | **38.2**
| Precision  | **68.1**                       | 66.0                           | 63.4                          | 49.6                             | 52.5                             | **54.9**
| Recall     | 74.2                           | **76.6**                       | 73.4                          | **59.5**                         | 56.9                             | 55.8



## Dataset variables, spatial and temporal resolutions

| Variables                                           | Spatial <br /> resolsution | Temporal <br /> resolution | MedST-28 <br /> pre-trained model | Fire Risk <br /> fine-tuned model | Fire Spread <br /> fine-tuned model |
|-----------------------------------------------------|:--------------------------:|:--------------------------:|:---------------------------------:|:---------------------------------:|:-----------------------------------:|
| Max Temperature                                     |9 x 9 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Max Wind Direction                                  |9 x 9 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Max Surface Pressure                                |9 x 9 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Min Relative Humidity                               |9 x 9 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Total Precipitation                                 |9 x 9 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Mean Surface Solar Radiation Downwards              |9 x 9 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Day Land Surface Temperature                        |1 x 1 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Night Land Surface Temperature                      |1 x 1 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Normalized Difference Vegetation Index (NDVI)       |500 x 500 m                 |    16 day                  | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Leaf Area Index (LAI)                               |500 x 500 m                 |    16 day                  | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Soil moisture                                       |5 x 5 km                    |    5 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Burned Areas                                        |1 x 1 km                    |    1 day                   | <b>✘</b>                          | <b>✘</b>                         | <b>✘</b> (Used as label)            |
| Ignition Points                                     |1 x 1 km                    |    1 day                   | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Slope                                               |30 x 30 m                   |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Aspect                                              |30 x 30 m                   |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Curvature                                           |30 x 30 m                   |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Population                                          |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of agriculture                             |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of forest                                  |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of grassland                               |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of settlements                             |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of shrubland                               |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of sparse vegetation                       |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of water bodies                            |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Fraction of wetland                                 |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |
| Roads distance                                      |300 x 300 m                 |    -                       | <b>✔</b>                          | <b>✔</b>                         | <b>✔</b>                            |

## Contact
For more info or to add a fine-tune task please contact: nikolas619065@gmail.com

## References
[https://github.com/sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE) <br/>
[https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae) <br />
[https://github.com/rwightman/pytorch-image-models/tree/master/timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm)
