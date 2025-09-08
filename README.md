# Project Ieroklis 
"Project Ieroklis" has combined two wildfire downstream tasks, Fire Risk and Fire Spread, by creating a small foundation model called MedST-28. From this model, two additional task-specific models were derived by retaining only the encoder part of MedST-28: an LSTM decoder was added for the Fire Risk task, and a Convolutional decoder was added for the Fire Spread task.

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

## Datasets
This repo contains two datasets for fine-tuning, but not the dataset used for the pre-training of the MedST-28 model, to provide more info please contact via email (nikolas619065@gmail.com).

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

For both fine-tuning datasets the stats are already calculated, which are used for normalization in the dataloader, if needed to be calculated run `calc_norm_values_MedST28.py` and in the first lines comment or uncomment lines 17 to 27 accordingly. 

## Pre-training
To pre-train the MedST-28 model



### Fine-tune
210 2310


## References
[https://github.com/sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE) <br/>
[https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae) <br />
[https://github.com/rwightman/pytorch-image-models/tree/master/timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm)
