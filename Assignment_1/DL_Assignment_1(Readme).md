Deep Learning Practical 1 – Fashion-MNIST
Problem Statement

Install and configure TensorFlow/Keras in Google Colab. Perform data preprocessing, normalization, train-test split, and visualization on a sample dataset.

Dataset

Fashion-MNIST

Fashion-MNIST is used instead of the original MNIST handwritten-digit dataset.

The dataset contains 10 classes of clothing/fashion images:

Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

Each image is 28 × 28 pixels.

Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn
Google Colab
Dataset Loading

The Fashion-MNIST dataset is loaded using TensorFlow/Keras:

import tensorflow as tf

(X, y), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

The images are reshaped from 28 × 28 to 784 pixels:

X = X.reshape(X.shape[0], 784)
Preprocessing

Pixel values originally range from 0 to 255. They are normalized to the range 0 to 1:

X = X.astype("float32") / 255.0
Train-Test Split

The dataset is divided into:

80% Training data
20% Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Expected shapes:

Training Images : (48000, 784)
Testing Images  : (12000, 784)
Training Labels : (48000,)
Testing Labels  : (12000,)
Visualization

The practical performs three visualizations:

A single Fashion-MNIST image.
Ten sample fashion images with their class names.
A histogram showing the distribution of the 10 Fashion-MNIST classes.
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
How to Run
1. Open Google Colab

Create a new Python notebook in Google Colab.

2. Import Required Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
3. Load the Dataset

Run the Fashion-MNIST loading cell.

4. Preprocess the Data

Run the reshape and normalization cells.

5. Split the Dataset

Run the train_test_split() cell.

6. Visualize the Data

Run the visualization cells to display sample images and class distribution.

Expected Result

The program should display:

Dataset shape: (60000, 784)
Labels shape: (60000,)
Training data: (48000, 784)
Testing data: (12000, 784)
Pixel value range: 0.0 to 1.0
Fashion-MNIST sample images
Class distribution histogram
Practical Objective

The practical demonstrates the basic deep-learning data pipeline:

Fashion-MNIST Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Reshaping
        ↓
Normalization
        ↓
Train-Test Split
        ↓
Visualization
Files
fashion_mnist_practical_1.ipynb
16_DL_Assignment_1_Fashion_MNIST.pdf
README.md
