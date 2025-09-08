# Project Ieroklis 
"Project Ieroklis" has combined two wildfire downstream tasks, Fire Risk and Fire Spread, by creating a small foundation model called MedST-28 which stands for Mediterranean Spatio-Temporal 28. <br />
From this model, two additional task-specific models were derived by retaining only the encoder part of MedST-28: an LSTM decoder was added for the Fire Risk task, and a Convolutional decoder was added for the Fire Spread task.

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

### MedST-28
The achitexcture used for the pre-training is shown below 

### MedST-28 fine-tune for Fire Risk


### MedST-28 fine-tune for Fire Spread


## Datasets
This repo contains two datasets for fine-tuning, but not the dataset used for the pre-training of the MedST-28 model, to provide more info please contact via email (nikolas619065@gmail.com).

| Variables       | MedST-28 | Fire Risk | Fire Spread |
|:---------------:|:---------:|:---------:|:-----------:|
| Max Temperature |            

### MedST-28 pre-train dataset
TODO

### Fire Risk fine-tune dataset
TODO

### Fire Spread fine-tune dataset
TODO

### Download links
| Dataset     | Link <br />(HuggingFace)                                                                                   | Size <br /> Zipped | Size <br /> Unzipped  |
|:-----------:|:----------------------------------------------------------------------------------------------------------:|:------------------:|:---------------------:|
| Fire Risk   | [Download](https://huggingface.co/datasets/nikos230/FireRisk/resolve/main/fire_risk_dataset_netcdf.zip)    | 148mb              | 812mb                 |
| Fire Spread | [Download](https://huggingface.co/datasets/nikos230/WildfireSpread/resolve/main/dataset_64_64_10days.zip)  | 12.1gb             | 31gb                  |

Fire Risk dataset was picked from Mesogeos datacube [Link to paper](https://arxiv.org/pdf/2306.05144), [Link to Github repo](https://github.com/orion-ai-lab/mesogeos) and Fire Spread dataset was picked from WildfireSpread forecasting with Deep Learning [Link to paper](https://arxiv.org/pdf/2505.17556), [Link to Github repo](https://github.com/nikos230/WildFireSpread) <br />

### Configure dataset paths
Download and put datasets (without changing their names) in the dataset folder, then all scripts for pre-training and fine-tune will run. If you put the dataset in another location you will have to configure their paths into the configs. <br />
- For the pre-training dataset open `configs/MedST28_dataset.yaml` and put the new path in line 4 and dataset stats file in line 5
- For Fire Risk dataset open `configs/FireRisk_dataset.yaml` and put the new path in line 4 and dataset stats file in line 5
- For Fire Spread dataset open `configs/WildfireSpread_dataset.yaml` and put the new path in line 4 and dataset stats file in line 5

For both fine-tuning datasets the stats are already calculated, which are used for normalization in the dataloader, if needed to be calculated again run `calc_norm_values_MedST28.py` and in the first lines comment or uncomment lines 17 to 27 accordingly. 

## Pre-training
To pre-train the MedST-28 model you need to download the pre-training dataset which has a size of ~500gb.
- Configure the checkpoint and visual results paths in `configs/pre-train_config.yaml` and chnage the hyper parameters if needed (like depth, patch size, mlp_ratio, etc..)
- Configure the MedST-28 dataset settings (time_steps, patch_size, variables to be included, train years, validation years, tests years, etc..) in the `configs/MedST28_dataset.yaml`
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
- Configure checkpoints and pre-trained model paths in `configs/fine-tune_config_fire_risk.yaml` in lines 5 and 6, you can also change epochs, loss function, patch_size and batch_size
- [Download](https://huggingface.co/datasets/nikos230/FireRisk/resolve/main/fire_risk_dataset_netcdf.zip) the Fire Risk dataset and configure paths as explained in ["Configure dataset paths"](Configure-dataset-paths)
- Finally run `fine-tune_MedST28_fire_risk.py`


### Fire Spread fine-tune
To fine-tune the MedST-28 model for fire spread you need a pre-train checkpoint of the MedST-28 model. Choose any of the checkpoints above, best results can be achived with MedST28 50M.
- Configure checkpoints and pre-trained model paths in `configs/fine-tune_config_fire_spread.yaml` in lines 5 and 6, you can also change epochs, loss function, patch_size and batch_size
- [Download](https://huggingface.co/datasets/nikos230/WildfireSpread/resolve/main/dataset_64_64_10days.zip) the Fire Spread dataset and configure paths as explained in ["Configure dataset paths"](Configure-dataset-paths)
- Finally run `fine-tune_MedST28_fire_spread.py`

## References
[https://github.com/sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE) <br/>
[https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae) <br />
[https://github.com/rwightman/pytorch-image-models/tree/master/timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm)
