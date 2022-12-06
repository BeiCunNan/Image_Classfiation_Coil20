# Coil20_Image_Classfiation
## Introduction

We built a total of six classic CNN network models such as LeNet5、AlexNet、GoogleNet、VGG16、ResNet50 and EfficientNet to achieve image classfication on COIL20.

### Dataset

The dataset we have put on Github, which has a total of 20 objects, each with 72 images, so there are 1440 images in total. The dimensions of each picture are [1,128,128].

Here I show you some of the data.

![Snipaste_2022-12-06_11-10-20](https://user-images.githubusercontent.com/105692522/205828070-c11f41a5-356d-4f22-8445-20cc115214ad.jpg)


### Result

We scramble the data first, and the divide the training and test sets by 9:1. The results are as follwos



## Requirement

- Python==3.9
- torch==1.11.0
- numpy==1.22.3
- transformers==4.19.2
- tqdm==4.64.0
- matplotlib==3.5.3
- pillow==9.1.0
- scikit-learn==1.1.1

## Preparation

### Clone

```shell
git clone https://github.com/BeiCunNan/Coil20_Image_Classfiation.git
```

### Create an anaconda environment

```shell
conda create -n cic python=3.9
conda activate cic
pip install -r requirements.txt
```

