# Dataset Description

## Overview
The Breast Ultrasound Images Dataset forms the cornerstone of our deep learning project, which focuses on the classification of breast ultrasound images into three distinct categories: normal, benign, and malignant. This dataset is instrumental in training and validating the ImageMaskNet, our custom CNN architecture, designed specifically for early breast cancer detection. The data is downloadable from https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

![Figure 2023-12-29 060030](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/87acda56-6d31-4e3e-af6b-15f7b142b292) 

_Figure 1: Comparative Display of Ultrasound Images with Segmentation Masks and Overlays for Benign and Malignant Breast Tissues. The left column illustrates benign cases with well-defined mask outlines, while the right column shows malignant cases with irregular mask patterns, both overlaid on the original images to emphasize the areas of interest._


## Composition and Source
The dataset comprises 780 ultrasound images, systematically collected from 600 female patients, ranging in age from 25 to 75 years. The data was gathered in 2018 with the primary objective of facilitating breast cancer research. Each image in the dataset maintains an average size of 500x500 pixels and is presented in PNG format, ensuring high-quality and consistency for computational analysis.

## Categories
The dataset categorizes images into three groups, each representing a different breast tissue condition:

- **Normal:** Images classified under this category depict healthy breast tissue, exhibiting no signs of tumors or other abnormalities.

- **Benign:** This category includes images showing non-cancerous tumors. Although these tumors are not immediately life-threatening, they require medical observation and are critical in understanding tumor development.

- **Malignant:** Images in this category indicate the presence of cancerous growths, highlighting cases that necessitate urgent medical attention.

## Significance
The dataset's diversity and comprehensiveness enable the effective training of deep learning models. It provides a realistic representation of various breast tissue conditions, facilitating the development of an AI system capable of distinguishing between normal, benign, and malignant states with high accuracy. The usage of this dataset in our project exemplifies the integration of machine learning in medical diagnostics, potentially revolutionizing the early detection and treatment of breast cancer.

# Methodology

## Introduction to Methodology
In this report, we explore the methodologies behind two specialized deep learning models developed for breast cancer detection using ultrasound imaging. Our investigation focuses on two distinct approaches to understand the efficacy and practicality of the models in real-world scenarios.

The journey began with Model 1, a convolutional neural network (CNN) designed exclusively for classification purposes. This model was trained and tested on a combination of ultrasound images and their corresponding masks, yielding impressive results. This initial success demonstrated the potential of combining mask and image data for accurate classification.

However, recognizing the practical limitations in clinical settings, where masks may not always be readily available, we developed Model 2. This model followed a similar training approach as Model 1, utilizing both mask and ultrasound data. The critical divergence came during the testing phase, where Model 2 was evaluated solely on ultrasound images, without the aid of masks. This approach was aimed at assessing the model's ability to leverage the learning from masks during training but still perform robustly when only ultrasound images are available for diagnosis.

The rationale behind this dual-model exploration is to evaluate the performance of a mask-informed CNN when tested under different conditions: one with the ideal scenario of having both masks and images, and the other simulating a more common clinical environment where only ultrasound images are available. This comparative study aims to provide insights into the adaptability and practical application of CNNs in breast cancer detection, bridging the gap between theoretical accuracy and clinical usability.

This section will detail each model's architecture, data preprocessing methods, training processes, and testing strategies, providing a comprehensive understanding of the methodologies employed in this research. Through this analysis, we aim to contribute valuable knowledge to the field of medical imaging and cancer diagnosis, emphasizing the importance of adaptable and realistic AI solutions in healthcare.

## Model 1: Dual-Input Classifier (DIC-Net)

### Model 1 Architecture: ImageMaskNet
ImageMaskNet is a dual-branch convolutional neural network (CNN) designed for analyzing breast ultrasound images along with their corresponding masks. The architecture is bifurcated into two distinct pathways – the image branch and the mask branch – each tailored to process a specific type of input.

![ImageMaskNet_FlowDiagram](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/252f653f-bc4e-4bc5-8734-0136b84a6982)

_Figure 2: Flow diagram of ImageMaskNet, illustrating the dual-pathway processing of RGB images and grayscale masks, feature concatenation, and classification into three categories._

## Image Branch:
- Input: This branch takes the standard 3-channel (RGB) ultrasound images.
- Layer 1: A convolutional layer with 32 filters, a kernel size of 3x3, and padding of 1. It is followed by a ReLU (Rectified Linear Unit) activation function.
- Pooling 1: A 2x2 max pooling layer reduces the spatial dimensions by half.
- Layer 2: Another convolutional layer with 64 filters, a 3x3 kernel, and padding of 1, followed by a ReLU activation function.
- Pooling 2: A second 2x2 max pooling layer further reduces the feature map size.

## Mask Branch:
- Input: This branch processes the single-channel (grayscale) mask images associated with the ultrasound data.
- Layer 1: A convolutional layer with 32 filters, 3x3 kernel size, and padding of 1, followed by a ReLU activation.
- Pooling 1: A 2x2 max pooling layer for spatial dimension reduction.
- Layer 2: A convolutional layer with 64 filters, 3x3 kernel, and padding of 1, followed by a ReLU activation function.
- Pooling 2: Another 2x2 max pooling layer.

## Combined Fully Connected Layers:
The ImageMaskNet architecture culminates in its combined fully connected layers, where features from the image and mask branches are flattened, merged, and then channeled through a dense neural layer of 128 neurons with ReLU activation. This integration harnesses the detailed insights from both the ultrasound imagery and masks. The neural network's final layer, reflecting the three distinct classification categories—normal, benign, and malignant—outputs the model's predictive verdict. This design aims to capitalize on the rich, complementary information from dual data sources to refine the model's diagnostic acumen for breast cancer detection.

## Data Preprocessing

Combining Multiple Masks
For some ultrasound images in our dataset, there were multiple mask files highlighting various areas of interest. To simplify the input to our neural network, we combined these multiple masks into a single mask image. This was done by overlaying the individual masks on top of each other, ensuring that no details were lost from the original set of masks.

![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/91eb57e4-5a0c-441e-bcc5-8047db5b30eb) ![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/d5c7c0d7-ba50-4db2-8349-d2bb9ae553c7) ![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/31b90fe3-7b58-45cf-9d40-b3a18c1b28b3) ![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/63b4530d-99a3-4e29-a58b-4e098560af20)

_Figure 2: The sequence of images demonstrates the process of combining multiple masks into one. The first image is the original ultrasound, followed by separate mask images, and ending with a single combined mask. This final, combined mask image is used for training the neural network, ensuring it has a complete view of all areas of interest._


The preprocessing of the Breast Ultrasound Images Dataset involved a series of transformational steps to render the images suitable for analysis by the ImageMaskNet model. Initially, each image within the dataset, irrespective of its classification as benign, malignant, or normal, was resized to a consistent dimension of 256x256 pixels to standardize the input size for the neural network.

A custom dataset handler, CustomImageMaskDataset, was utilized to streamline the loading and processing of the images alongside their associated masks. The image transformation pipeline incorporated a resizing step followed by a conversion to tensor format, facilitating compatibility with the PyTorch framework used for model training.

For the training subset, data augmentation was applied, introducing random horizontal flips to the images. This technique was intended to diversify the training data, aiding the model in developing robustness against variations and potentially enhancing its generalization capabilities.

## Training

Model Evaluation and Validation:

Challenges and Solutions:

















Address any challenges encountered during the model development and training process.

Discuss the solutions or adjustments made to overcome these challenges.










## Citation
In acknowledgment of the dataset's origin and in compliance with academic standards, we cite the following source for the dataset used in our project:
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
