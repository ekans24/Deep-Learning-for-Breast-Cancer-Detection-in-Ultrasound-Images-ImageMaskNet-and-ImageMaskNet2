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

## Data Preprocessing

#### Combining Multiple Masks
For some ultrasound images in our dataset, there were multiple mask files highlighting various areas of interest. To simplify the input to our neural network, we combined these multiple masks into a single mask image. This was done by overlaying the individual masks on top of each other, ensuring that no details were lost from the original set of masks.

The design of ImageMaskNet is predicated on the synthesis of domain expertise in medical imaging and established practices in machine learning. The architecture is meticulously constructed to capitalize on the complementary information provided by both ultrasound images and segmentation masks, facilitating a comprehensive approach to the classification task at hand. The aforementioned architectural decisions are made with the intent to enhance diagnostic accuracy, ensuring the model's utility in a clinical setting.

![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/91eb57e4-5a0c-441e-bcc5-8047db5b30eb) ![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/d5c7c0d7-ba50-4db2-8349-d2bb9ae553c7) ![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/31b90fe3-7b58-45cf-9d40-b3a18c1b28b3) ![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/63b4530d-99a3-4e29-a58b-4e098560af20)

_Figure 3: The sequence of images demonstrates the process of combining multiple masks into one. The first image is the original ultrasound, followed by separate mask images, and ending with a single combined mask. This final, combined mask image is used for training the neural network, ensuring it has a complete view of all areas of interest._


The preprocessing of the Breast Ultrasound Images Dataset involved a series of transformational steps to render the images suitable for analysis by the ImageMaskNet model. Initially, each image within the dataset, irrespective of its classification as benign, malignant, or normal, was resized to a consistent dimension of 256x256 pixels to standardize the input size for the neural network.

A custom dataset handler, CustomImageMaskDataset, was utilized to streamline the loading and processing of the images alongside their associated masks. The image transformation pipeline incorporated a resizing step followed by a conversion to tensor format, facilitating compatibility with the PyTorch framework used for model training.

For the training subset, data augmentation was applied, introducing random horizontal flips to the images. This technique was intended to diversify the training data, aiding the model in developing robustness against variations and potentially enhancing its generalization capabilities.

## Model 1

### ImageMaskNet Architecture Rationalization

#### Architectural Overview
ImageMaskNet is a specialized convolutional neural network designed to process and classify breast ultrasound images. The architecture consists of two distinct branches: the image branch for processing RGB images and the mask branch for processing grayscale segmentation masks. Each branch is tailored to extract features from its respective input type, which are then combined to inform the classification decision. This section delineates the rationale behind the design choices in ImageMaskNet's architecture.

![ImageMaskNet_FlowDiagram](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/252f653f-bc4e-4bc5-8734-0136b84a6982)

_Figure 2: Flow diagram of ImageMaskNet, illustrating the dual-pathway processing of RGB images and grayscale masks, feature concatenation, and classification into three categories._

#### Image Branch:
- Input: This branch takes the standard 3-channel (RGB) ultrasound images.
- Layer 1: A convolutional layer with 32 filters, a kernel size of 3x3, and padding of 1. It is followed by a ReLU (Rectified Linear Unit) activation function.
- Pooling 1: A 2x2 max pooling layer reduces the spatial dimensions by half.
- Layer 2: Another convolutional layer with 64 filters, a 3x3 kernel, and padding of 1, followed by a ReLU activation function.
- Pooling 2: A second 2x2 max pooling layer further reduces the feature map size.

#### Convolutional Layers
The image branch begins with convolutional layers, which are fundamental to CNNs for their ability to perform effective spatial feature extraction. The initial layer comprises 32 filters with a kernel size of 3x3 and padding of 1. This configuration enables the network to detect basic visual elements such as edges and textures. The subsequent layer, consisting of 64 filters, allows the network to construct more complex representations from these elementary features. The increase in filter count is a deliberate strategy to enhance the model's capacity to capture a wider variety of features at increasing levels of abstraction.

#### Activation Functions
The activation function employed after each convolutional layer is the Rectified Linear Unit (ReLU). ReLU is selected for its efficiency in accelerating the convergence of stochastic gradient descent compared to sigmoid or tanh functions. Moreover, ReLU helps mitigate the issue of vanishing gradients, enabling deeper networks to be trained more effectively.

#### Pooling Layers
Max pooling layers follow each convolutional layer, utilizing a 2x2 window to downsample the feature maps by a factor of two. This downsampling serves two primary functions: it reduces the computational load for subsequent layers and introduces a level of translational invariance to the learned features, thereby improving the robustness of the model.

## Mask Branch:
- Input: This branch processes the single-channel (grayscale) mask images associated with the ultrasound data.
- Layer 1: A convolutional layer with 32 filters, 3x3 kernel size, and padding of 1, followed by a ReLU activation.
- Pooling 1: A 2x2 max pooling layer for spatial dimension reduction.
- Layer 2: A convolutional layer with 64 filters, 3x3 kernel, and padding of 1, followed by a ReLU activation function.
- Pooling 2: Another 2x2 max pooling layer.

#### Single-Channel Input
The mask branch processes segmentation masks, which are images that mark specific regions in ultrasound scans. It's structured similarly to the image branch, using the same settings for the layers, which ensures the network treats both image and mask data consistently. This setup allows the model to learn important features from both the detailed images and the masks effectively.
  
### Combined Fully Connected Layers:

#### Feature Integration
Subsequent to feature extraction, the image and mask branches converge, and the extracted features are concatenated. This concatenated feature vector is then passed through fully connected layers. A dense layer of 128 neurons is employed, chosen to provide a balance between model expressiveness and computational efficiency. The use of a fully connected layer here is critical for integrating the features from both the image and mask pathways, which is hypothesized to enhance the model's predictive capability.

#### Classification Output
The architecture concludes with a softmax output layer that categorizes the combined features into three classes: normal, benign, and malignant. The softmax layer is the industry standard for multi-class classification problems due to its ability to output a normalized probability distribution over predicted classes.

## Training

In the development of ImageMaskNet, the dataset was partitioned into two subsets: 80% was allocated for training the model, while the remaining 20% was reserved for testing its performance. This approach ensured that a substantial amount of data was used for the model to learn the underlying patterns, while still retaining a separate dataset to unbiasedly evaluate its generalization capabilities.

The training phase was executed over 10 epochs, with each epoch encapsulating a complete iteration across the entire training dataset. The learning rate and weight decay parameters were fine-tuned to 0.001 and 1e-4, respectively, as determined by a thorough grid search to ascertain the optimal settings. These parameters play a crucial role in guiding the convergence of the model towards a robust solution that avoids overfitting.

Throughout the training process, epoch—loss and accuracy and accuracy were recorded. The loss metric gauges the model's prediction errors, with a lower loss indicative of improved model predictions. Conversely, accuracy measures the proportion of correctly predicted instances, providing a direct reflection of the model's predictive prowess. The overarching aim was to observe a consistent reduction in loss alongside an increment in accuracy, signifying the model's successful learning trajectory.

## Model Evaluation and Validation:

An integral part of our model evaluation is the visual validation of predictions made by ImageMaskNet. 

![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/171c38b6-1d65-4602-b8c0-a8fe144f1e07)

_Figure 4: illustrates a selection of ultrasound images across the three classes—benign, malignant, and normal—with both the ground truth and the ImageMaskNet's predictions labeled._


### Performance Overview
The ImageMaskNet was subjected to a rigorous training and testing process over 10 epochs. The performance metrics, as depicted in the accompanying graphs, provide a detailed insight into the model's behavior over time.

### Accuracy Analysis
The accuracy graph shows a clear trend of increasing accuracy on both the training and testing datasets as the epochs progress (Figure X). The model's training accuracy started at 65.9% and improved significantly to 99.36% by the 10th epoch. The testing accuracy followed a similar upward trend, starting at 71.79% and reaching up to 98.72%. The close convergence of training and testing accuracies indicates that the model generalizes well and is not merely memorizing the training data.

![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/2a329eb6-ca0c-4522-9ef5-e757f93bac20)

_Figure 5 - Accuracy vs. Epoch: A line graph showing the training and test accuracy over the epochs._

### Loss Analysis
The loss graph complements the accuracy analysis by showing a corresponding decrease in loss for both the training and testing data (Figure Y). The training loss reduced from 0.9322 to 0.0278, and the testing loss followed suit, starting from 0.5944 and ending at 0.0468. The convergence of training and test loss also suggests that the model is learning general features rather than overfitting to the training set.

![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/78a3a542-1fef-481b-a5e2-741431b0a792)

_Figure 6 - Loss vs. Epoch: A line graph illustrating the decline in training and test loss over the epochs._

### Precision and Recall
In addition to accuracy, precision and recall are critical metrics, particularly in the medical imaging domain where the cost of false positives and negatives can be high. The model achieved a final test precision and recall of 98.74% and 98.72%, respectively. These values are highly desirable, showing that the model is not only accurate but also reliable in its positive predictions (precision) and sensitive in identifying positive cases (recall).

### Summary of Results
The following table summarizes the key performance metrics at the final epoch:

| Metric  | 	Value |
| ------------- | ------------- |
| Train Accuracy | 99.36%  |
| Test Accuracy  | 98.72%  |
| Train Loss  | 0.0278  |
| Test Loss | 0.0468  |
| Precision  | 98.74%  |
| Recall | 98.72%  |

The high values across all metrics underscore the robustness of the ImageMaskNet in classifying breast ultrasound images.

### Interpretation
The plots and metrics tell a story of a model that learns efficiently and generalizes well to unseen data. The rapid increase in accuracy and decrease in loss during the initial epochs suggest that the model is capable of quickly assimilating the patterns within the dataset. The plateauing of both accuracy and loss in later epochs indicates that the model may have reached its learning capacity given the current architecture and dataset.

It's noteworthy that despite the high accuracy, the model did not reach a perfect score. This is a realistic outcome, reflecting the inherent uncertainty and variability in medical image interpretation.

## Model 2

### ImageMaskNet2 Architecture Rationalization
ImageMaskNet2 embodies a sophisticated dual-pathway architecture, uniquely crafted to harness the complementary features of ultrasound imagery and associated segmentation masks during training. This innovative design is pivotal for enriching the model's feature extraction capabilities. However, its true novelty lies in its operational versatility during inference, where it seamlessly transitions to an image-only mode, maintaining robust performance even in the absence of mask data.

![ImageMaskNet_Training_Mode_FlowDiagram](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/ee303906-6678-4fd3-8c0c-0166f9fea0dd)

_Figure 7: ImageMaskNet's architecture, showcasing dual-input convolutional branches for image and mask data that merge for classification into 'normal', 'benign', or 'malignant' categories, with an image-only inference path when masks are not provided._

### Enhanced Image Branch
The image branch is the backbone of ImageMaskNet2, featuring a series of convolutional layers with increasing depth—32, 64, and 128 filters. This hierarchical design is deliberate, ensuring the capture of complex features from simple to intricate. Each convolutional layer is fortified with batch normalization and ReLU activation, establishing a stable and efficient learning trajectory. Max pooling layers interspersed between convolutional layers serve a dual purpose: they compact the feature representation and endow the network with translational invariance, essential for focusing on pertinent features within the ultrasound images.

- **Conv2d Layers:** These layers extract a hierarchy of features, from basic edges and textures to more complex patterns.
- **BatchNorm2d:** Normalization steps stabilize the learning process, accelerate convergence, and have been shown to improve overall network performance.
- **ReLU:** The non-linear activation function introduces non-linearity to the learning process, enabling the network to learn complex mappings between input data and labels.
- **MaxPool2d:** Pooling layers reduce dimensionality, condense feature representations, and imbue the network with a degree of translational invariance.

### Mask Branch with Regularization
The mask branch is ImageMaskNet2's strategic component, processing grayscale masks to accentuate critical regions within the images. Structurally parallel to the image branch, it integrates a dropout layer post the convolutional sequence, strategically set at a 50% rate to deter overfitting. This branch's output does not directly contribute to the inference output but is indispensable during training, offering a regularization effect that enhances the generalization capacity of the network.

- **Dropout:** A dropout rate of 50% is employed to prevent over-reliance on any particular neuron within the network, encouraging a more robust feature representation.

### Combined Fully Connected Layers
Post feature extraction, ImageMaskNet2 diverges into two potential paths: a combined mode and an image-only mode. In combined mode, features from both branches merge, traversing through dense layers that synthesize the information into a cohesive feature vector for classification. The image-only mode allows for autonomous operation of the image branch, directly mapping its features to the output classes. This bifurcation is pivotal for the model's adaptability, ensuring operational efficacy in diverse clinical scenarios.

- **Linear:** The linear layers map the integrated features to the space of the output classes.

- **Output Classes:** The network concludes with a softmax output layer, providing probabilistic interpretations for each class in a multi-class classification setting.

### Output Classification
Concluding its architecture, ImageMaskNet2 employs a softmax layer that articulates the probability distribution across the potential classes—normal, benign, and malignant. This layer is the culmination of the model's intricate processing, offering a probabilistic understanding of the classification decision.

### Training and Testing Flexibility
ImageMaskNet2's dual-mode functionality is emblematic of its design ingenuity. It is trained dually yet possesses the agility to infer solely from image data, a feature that greatly enhances its clinical applicability. The network is trained to integrate and amplify the signal from the mask data, but when deployed, it is capable of delivering diagnostic predictions with just the image data, a common scenario in clinical environments.

### Final Notes on ImageMaskNet2
The architecture of ImageMaskNet2 stands as a tribute to the advancement in neural network architectures, adeptly balancing computational depth with practical applicability. It exemplifies a progressive approach to medical image analysis, underlined by a design that is both flexible and robust, ensuring that the network remains versatile across training and deployment phases.

## Training

## Model Evaluation and Validation:

Here is an some visuals from the predicted classes of the model. Three images from each class were selected.

![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/def72c0c-5c7b-4945-89e9-a047259818f4)

_Figure 8: illustrates a selection of ultrasound images across the three classes—benign, malignant, and normal—with both the ground truth and the ImageMaskNet2's predictions labeled._

### Performance Overview
The ImageMaskNet was subjected to training and testing process over 50 epochs. This model needed more epochs than the previous one to reach a convergence and stable values. The performance metrics, as depicted in the accompanying graphs, provide a detailed insight into the model's behavior over time.

### Accuracy Analysis
![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/e079d4b7-f2b3-41bb-92d6-0ebe6483ae42)

### Loss Analysis
![image](https://github.com/ekans24/Breast-Cancer-Detection-with-ImageMaskNet-CNN/assets/93953899/2d61a61b-1d07-4bb3-a359-fe3b8e60441f)

### Precision and Recall

### Summary of Results
The following table summarizes the key performance metrics at the final epoch:

 final: Test Loss: 0.1135 - Test Accuracy: 0.9692

 Loss: 0.2638 - Accuracy: 0.9077 - Precision: 0.9054 - Recall: 0.8931

| Metric  | 	Value |
| ------------- | ------------- |
|  |  |
|   |  |
|   |   |
|  |   |
|  |   |
| |  |

## Compare and Contrast 

## Discussions

## Limitations

## Future Work

## Conclusion

## Citation
In acknowledgment of the dataset's origin and in compliance with academic standards, we cite the following source for the dataset used in our project:
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
