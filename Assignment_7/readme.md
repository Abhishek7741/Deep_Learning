# Assignment 07 – Transfer Learning for Image Classification

## 📌 Aim

To implement transfer learning using pretrained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for image classification using the CIFAR-10 dataset and compare their performance.

## 📝 Problem Statement

Implement transfer learning using pretrained CNN architectures and evaluate their performance using test accuracy, test loss, training time, confusion matrices, and classification reports.

## 📊 Dataset Description

The project uses the **CIFAR-10 dataset**, which contains 60,000 color images belonging to 10 different classes.

Each original image has a resolution of 32 × 32 pixels with three color channels (RGB).

### CIFAR-10 Classes

| Label | Class |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

### Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training samples used | 5,000 |
| Training samples after split | 4,000 |
| Validation samples | 1,000 |
| Test samples | 1,000 |
| Image size | 224 × 224 |
| Number of classes | 10 |
| Batch size | 16 |
| Random state | 42 |

The notebook uses a subset of the dataset to reduce computational and memory requirements.

## 🧠 What Is Transfer Learning?

Transfer learning is a deep learning technique in which a model pretrained on a large dataset is reused for a related task.

Instead of training a CNN from scratch, the pretrained model provides previously learned features such as edges, corners, textures, shapes, and object structures.

In this assignment, ImageNet-pretrained models are adapted to classify CIFAR-10 images into 10 categories.

## 🤖 Models Used

### 1. AlexNet

AlexNet is a convolutional neural network containing convolutional layers, ReLU activations, max-pooling layers, fully connected layers, and dropout.

In this experiment, the pretrained weights are frozen and the final classification layer is replaced with a 10-class output layer.

### 2. VGG16

VGG16 is a deep CNN developed by the Visual Geometry Group at the University of Oxford.

It contains 13 convolutional layers and 3 fully connected layers. Its pretrained weights are used to extract image features, while the final classifier is modified for CIFAR-10.

### 3. ResNet50

ResNet50 is a deep residual network that uses skip connections to improve information and gradient flow through the network.

The pretrained model is adapted by replacing its final fully connected layer with a classifier containing 10 output neurons.

### 4. EfficientNetB0

EfficientNetB0 is designed to balance classification accuracy, model size, and computational cost.

Its pretrained feature extractor is retained, and its final classification layer is replaced with a 10-class classifier.

## ⚙️ Image Preprocessing

The following preprocessing steps are performed:

1. Download CIFAR-10 using Torchvision.
2. Resize images from 32 × 32 to 224 × 224 pixels.
3. Apply random horizontal flipping during training.
4. Convert images into PyTorch tensors.
5. Normalize RGB channels using ImageNet mean and standard deviation.
6. Split the selected training subset into training and validation datasets.
7. Load data in batches using PyTorch DataLoaders.

### ImageNet Normalization

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## 🏗️ Transfer Learning Workflow

```text
CIFAR-10 Dataset
        ↓
Image Preprocessing
        ↓
Train / Validation / Test Split
        ↓
Load ImageNet-Pretrained Model
        ↓
Freeze Pretrained Parameters
        ↓
Replace Final Classification Layer
        ↓
Train New Classifier
        ↓
Evaluate on Test Dataset
        ↓
Compare Four Models
```

## ⚙️ Model Configuration

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 |
| Batch size | 16 |
| Number of epochs | 3 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross Entropy Loss |
| Output classes | 10 |
| Pretrained weights | ImageNet |
| Trainable parameters | Newly added classifier layer |

The pretrained parameters are frozen, and only the newly added classifier parameters are trained.

## 🚀 Methodology

1. Import PyTorch, Torchvision, NumPy, Pandas, and evaluation libraries.
2. Select CUDA GPU if available; otherwise, use CPU.
3. Download the CIFAR-10 dataset.
4. Resize and normalize images.
5. Create training, validation, and test DataLoaders.
6. Load AlexNet, VGG16, ResNet50, and EfficientNetB0 with pretrained ImageNet weights.
7. Freeze the pretrained model parameters.
8. Replace each model's final classification layer with a 10-class classifier.
9. Train each model for three epochs.
10. Track training and validation loss and accuracy.
11. Restore the weights with the best validation accuracy.
12. Evaluate each model on the test dataset.
13. Compare test accuracy, test loss, and training time.
14. Generate accuracy and loss graphs.
15. Generate confusion matrices and classification reports.
16. Identify the best-performing and fastest models.

## 📈 Evaluation Metrics

- **Test Accuracy:** Percentage of correctly classified
