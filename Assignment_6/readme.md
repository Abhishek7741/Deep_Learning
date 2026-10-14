# Assignment 06 – CNN-Based Corn/Maize Leaf Disease Classification

## 📌 Aim

To design and implement a Convolutional Neural Network (CNN) for image classification and identify diseases in corn/maize leaves using deep learning.

## 📝 Problem Statement

Design and implement a Convolutional Neural Network (CNN) for image classification using a plant disease dataset.

This practical uses the **Corn or Maize Leaf Disease Dataset** to classify leaf images into healthy and diseased categories.

## 📊 Dataset Description

The project uses the Corn or Maize Leaf Disease Dataset from Kaggle.

The dataset contains images of corn/maize leaves belonging to four classes.

### Classes

1. **Blight**
2. **Common Rust**
3. **Gray Leaf Spot**
4. **Healthy**

### Dataset Information

| Parameter | Value |
|---|---|
| Dataset | Corn/Maize Leaf Disease Dataset |
| Image size | 128 × 128 pixels |
| Training split | 80% |
| Validation split | 20% |
| Batch size | 32 |
| Number of classes | 4 |
| Total images | 4,188 |
| Training images | 3,351 |
| Validation images | 837 |

The dataset is downloaded automatically using KaggleHub.

## 🧠 What Is CNN?

A Convolutional Neural Network (CNN) is a deep learning architecture commonly used for image classification and computer vision tasks.

CNNs automatically extract useful visual features such as edges, textures, shapes, spots, and patterns from images.

In this project, the CNN learns visual characteristics of corn/maize leaves to distinguish between different diseases and healthy leaves.

## ⚙️ Image Preprocessing

The following preprocessing steps are performed:

1. Download the dataset from Kaggle.
2. Load images from class-specific directories.
3. Resize all images to 128 × 128 pixels.
4. Divide the dataset into training and validation subsets.
5. Normalize pixel values from the range 0–255 to 0–1.
6. Load images in batches of 32.
7. Cache and prefetch datasets to improve processing efficiency.

## 🏗️ CNN Model Architecture

The CNN consists of the following layers:

```text
Input Image: 128 × 128 × 3
             ↓
      Rescaling Layer
      Pixel Values: 0–1
             ↓
    Conv2D – 32 Filters
             ↓
       Max Pooling
             ↓
    Conv2D – 64 Filters
             ↓
       Max Pooling
             ↓
   Conv2D – 128 Filters
             ↓
       Max Pooling
             ↓
          Flatten
             ↓
    Dense – 128 Neurons
             ↓
       Dropout – 0.5
             ↓
    Dense – 4 Neurons
      Softmax Activation
             ↓
   Predicted Disease Class
```

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | CNN |
| Convolutional layers | 3 |
| Convolution filters | 32, 64, 128 |
| Kernel size | 3 × 3 |
| Pooling | MaxPooling2D |
| Dense layer | 128 neurons |
| Dropout rate | 0.5 |
| Output classes | 4 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Sparse Categorical Crossentropy |
| Evaluation metric | Accuracy |
| Epochs | 10 |
| Batch size | 32 |

## 🚀 Methodology

The project follows these steps:

1. Import the required libraries.
2. Download the Corn/Maize Leaf Disease Dataset.
3. Load and inspect the images.
4. Create training and validation datasets.
5. Resize and normalize images.
6. Build the CNN architecture.
7. Compile the model using Adam optimizer and sparse categorical cross-entropy loss.
8. Train the CNN for 10 epochs.
9. Record training and validation accuracy and loss.
10. Generate predictions on validation images.
11. Create a confusion matrix.
12. Generate a classification report containing precision, recall, and F1-score.
13. Visualize the training results and sample leaf images.

## 📈 Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy:** Percentage of correctly classified images.
- **Precision:** Measures the proportion of predicted class labels that are correct.
- **Recall:** Measures the proportion of actual class images correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each disease class.
- **Training and Validation Loss:** Measures classification error during training and validation.

## 📊 Results

According to the practical execution, the model achieved the following validation results:

| Metric | Result |
|---|---:|
| Validation Accuracy | 83% |
| Weighted Precision | 84% |
| Weighted Recall | 83% |
| Weighted F1-score | 81% |
| Validation Images | 837 |

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Blight | 0.64 | 0.92 | 0.75 |
| Common Rust | 0.97 | 0.91 | 0.94 |
| Gray Leaf Spot | 0.64 | 0.19 | 0.29 |
| Healthy | 0.98 | 1.00 | 0.99 |

The model performs particularly well in identifying healthy leaves and Common Rust. However, the low recall for Gray Leaf Spot indicates that this disease is frequently misclassified.

## 📉 Visualizations

The notebook generates the following visualizations:

### 1. Sample Dataset Images

Displays sample images from the four classes: Blight, Common Rust, Gray Leaf Spot, and Healthy.

### 2. Training and Validation Accuracy

Shows how training and validation accuracy change over the 10 epochs.

### 3. Training and Validation Loss

Displays the change in training and validation loss during model training.

### 4. Confusion Matrix

Visualizes correct and incorrect predictions for each of the four disease categories.

### 5. Classification Report

Displays precision, recall, F1-score, and support for each class.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub
- Jupyter Notebook / Google Colab

## 📦 Installation

Install the required libraries:

```bash
pip install tensorflow kagglehub numpy matplotlib seaborn scikit-learn
```

## ▶️ How to Run

1. Open Jupyter Notebook or Google Colab.
2. Open your Assignment 6 notebook.
3. Run the cells sequentially.
4. Allow KaggleHub to download the dataset.
5. Train the CNN model.
6. Review the accuracy and loss graphs, confusion matrix, and classification report.

The dataset is downloaded automatically using:

```python
import kagglehub

path = kagglehub.dataset_download(
    "smaranjitghose/corn-or-maize-leaf-disease-dataset"
)
```

## 📁 Project Structure

```text
Assignment-6/
│
├── Assignment-6.ipynb
└── README.md
```

*Replace `Assignment-6.ipynb` with the actual notebook filename if it differs.*

## ✅ Conclusion

This practical demonstrates the use of a Convolutional Neural Network for automated corn/maize leaf disease classification.

The CNN achieved **83% validation accuracy** across four classes: Blight, Common Rust, Gray Leaf Spot, and Healthy. The results show strong classification performance for healthy leaves and Common Rust, while Gray Leaf Spot remains more challenging to identify.

The experiment demonstrates how deep learning can extract visual features from leaf images and assist in plant disease classification.

---

**Author:** Abhishek Dasharath Kadam  
**Subject:** Deep Learning  
**Practical No.:** 6  
**Department:** CSE-AI  
**Institution:** Vishwakarma Institute of Technology, Pune
