# Project Ieroklis 
Ο Ιεροκλής ο Αλεξανδρινός (αρχαία ελληνικά: Ἱεροκλῆς ὁ Ἀλεξανδρεύς‎‎) ήταν Έλληνας Νεοπλατωνικός φιλόσοφος. Δίδαξε στην Αλεξάνδρεια από το 420 μέχρι και το 450 μ.Χ. 
Οι θεωρίες του είναι βασισμένες στον πλατωνισμό, ενώ έχει κι εμφανείς επιρροές από τον Αριστοτέλη, τους στωικούς και το χριστιανισμό.

## About
This repo contains ready-to-use python scripts to:
- **Pre-train** Masked Auto Encoders (MAE) with Vision Transformer (ViT) backbone
- **Fine-tune** pre-trained models with MAE encoder and Convolutional decoder

## Getting Started
First create a new conda enviroment, with any name, here name was choosen **mae_env**
```
conda create -n mae_env 
```
After installation activate the new enviroment, replace "mae_env" with your enviroment name
```
conda activate mae_env
```
Finally install requirements.txt
```
pip install -r requirements.txt
```
Install PyTorch separately, to avoid any errors
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Dataset
This repo was created for multi-spectral remote sensing images. Pre-training script can run with any image size (etc. 224x224, 64x64, 32x32, ...) and with any given number of channels. 
Data is splitted into 2 folders, **images** and **masks**. The fist is used for pre-training and both folders are used for fine-tune. Pre-trainng images are splitted into pre-training and validation, this is done with the date of the file names etc. "20250512" where 2025 is the year, 08 is the day of the month and the 05 is the month

## Pre-training
With this repo you can use any set of images to pre-train the Masked Auto Encoder ViT model. Given your data is seperated into 2 folders, images and masks. For pre-training only images are needed. In this repo the provided dataloader is configured to split pre-train and validation (and test) sets with the date of the images, example: images from 2022 to 2024 are used for pre-train, images from 2021 are used for validation

### Dataloader

### Fine-tune
210 2310


## References
git repo go here
