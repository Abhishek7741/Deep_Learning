# Deep Learning Practical 1 – Fashion-MNIST

## 📌 Problem Statement

Install and configure TensorFlow/Keras in Google Colab. Perform data preprocessing, normalization, train-test split, and visualization on a sample dataset.

---

## 📊 Dataset

This practical uses the **Fashion-MNIST** dataset.

Fashion-MNIST contains 70,000 grayscale images of fashion items. Each image has a size of **28 × 28 pixels** and belongs to one of 10 classes.

### Classes

| Label | Class |
|------:|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

## 🔄 Workflow

The practical follows these steps:

```text
Fashion-MNIST Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Image Reshaping
        ↓
Normalization
        ↓
Train-Test Split
        ↓
Data Visualization
        ↓
Class Distribution

📥 Dataset Loading

The Fashion-MNIST dataset is loaded using TensorFlow/Keras.
import tensorflow as tf

(X, y), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
The images are reshaped from 28 × 28 into 784 features:
X = X.reshape(X.shape[0], 784)
🔧 Data Preprocessing

The pixel values are originally in the range 0–255.

They are normalized to the range 0–1:
X = X.astype("float32") / 255.0
Normalized Pixel Range
Minimum Pixel Value: 0.0
Maximum Pixel Value: 1.0
✂️ Train-Test Split

The dataset is divided into:

80% Training Data
20% Testing Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
Dataset Shapes
Total Images    : (60000, 784)
Total Labels    : (60000,)

Training Images : (48000, 784)
Testing Images  : (12000, 784)

Training Labels : (48000,)
Testing Labels  : (12000,)
📈 Visualization

The practical performs the following visualizations:

1. Sample Image

A single Fashion-MNIST image is displayed using Matplotlib.

2. Multiple Sample Images

Ten Fashion-MNIST images are displayed with their corresponding class names.

3. Class Distribution

A histogram is plotted to show the distribution of the 10 Fashion-MNIST classes.

👕 Class Names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
▶️ How to Run
Step 1: Open Google Colab

Open Google Colab and create a new Python notebook.

Step 2: Install/Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
Step 3: Load Fashion-MNIST
(X, y), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
Step 4: Reshape the Images
X = X.reshape(X.shape[0], 784)
Step 5: Normalize the Data
X = X.astype("float32") / 255.0
Step 6: Split the Dataset

Run the train_test_split() code.

Step 7: Visualize the Dataset

Run the visualization code to display sample images and the class distribution.

📁 Project Structure
Deep-Learning-Practical-1/
│
├── fashion_mnist_practical_1.ipynb
├── 16_DL_Assignment_1_Fashion_MNIST.pdf
└── README.md
🎯 Learning Objectives

After completing this practical, we learn how to:

Load a dataset using TensorFlow/Keras.
Inspect image and label data.
Reshape image data.
Normalize pixel values.
Split data into training and testing sets.
Visualize image samples.
Analyze class distribution.
Prepare image data for deep learning models.
✅ Expected Output

The program should display:

Images Shape: (60000, 784)
Labels Shape: (60000,)

Minimum Pixel Value: 0.0
Maximum Pixel Value: 1.0

Training Data: (48000, 784)
Testing Data : (12000, 784)

Training Images : (48000, 784)
Testing Images  : (12000, 784)

Training Labels : (48000,)
Testing Labels  : (12000,)

Pixel Value Range : 0.0 to 1.0

It should also display:

Fashion-MNIST sample images
Ten clothing categories
Fashion-MNIST class distribution histogram
