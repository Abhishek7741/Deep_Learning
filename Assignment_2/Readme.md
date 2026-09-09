# Student Performance Prediction Using MLP

##  Project Overview

This project uses a **Multi-Layer Perceptron (MLP)** neural network to predict a student's academic grade based on their study habits, attendance, class participation, and total score.

The project demonstrates the use of **Machine Learning and Deep Learning concepts** for multi-class student performance classification.

---

## Objective

The main objective of this project is to predict the student's grade category:

* A
* B
* C
* D
* F

based on the following performance-related features:

* Weekly self-study hours
* Attendance percentage
* Class participation
* Total score

---

##  Dataset

The project uses a dataset named:

`student_performance.csv`

The dataset contains **190,362 records** and **6 columns**.

### Dataset Columns

| Column                    | Description                             |
| ------------------------- | --------------------------------------- |
| `student_id`              | Unique student identifier               |
| `weekly_self_study_hours` | Number of hours spent studying per week |
| `attendance_percentage`   | Student attendance percentage           |
| `class_participation`     | Level of participation in class         |
| `total_score`             | Overall student score                   |
| `grade`                   | Student's final grade                   |

### Grade Distribution

| Grade | Records |
| ----- | ------: |
| A     | 104,489 |
| B     |  49,287 |
| C     |  26,808 |
| D     |   8,565 |
| F     |   1,213 |

The notebook creates a balanced sample by selecting up to **20,000 records per grade**. This results in **69,778 samples**.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* MLP Neural Network

---

##  Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

##  Methodology

The project follows these steps:

### 1. Load Dataset

The `student_performance.csv` file is loaded using Pandas.

### 2. Select Features

Four features are selected:

```text
weekly_self_study_hours
attendance_percentage
class_participation
total_score
```

The target variable is:

```text
grade
```

### 3. Balanced Sampling

The dataset is grouped according to grade and a maximum of **20,000 samples per grade** is selected.

This helps reduce the effect of the highly imbalanced original grade distribution.

### 4. Encode Grade Labels

The categorical grades are converted into numerical labels using `LabelEncoder`.

The classes are:

```text
A, B, C, D, F
```

### 5. Train-Test Split

The sampled dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified splitting is used to maintain the class distribution.

### 6. Feature Scaling

`StandardScaler` is used to standardize the input features.

The scaler is fitted only on the training data and then applied to the test data.

### 7. MLP Model

An MLP classifier is created with:

* Hidden layers: `(64, 32)`
* Activation: `ReLU`
* Optimizer: `Adam`
* L2 regularization: `0.0001`
* Early stopping: Enabled
* Validation fraction: `15%`
* Maximum iterations: `100`
* Batch size: `256`

### 8. Model Training

The model is trained on the scaled training data.

The model stopped after **47 epochs** because early stopping was enabled.

### 9. Prediction and Evaluation

The trained model predicts grades for the test dataset.

The following evaluation metrics are used:

* Accuracy
* Confusion Matrix
* Classification Report

---

## Results

The model achieved:

### Accuracy

**99.73%**

```text
Accuracy: 0.9972771567784465
Accuracy Percentage: 99.73%
```

The dataset was divided into:

* Training samples: **55,822**
* Testing samples: **13,956**

### Confusion Matrix

```text
[[3991    9    0    0    0]
 [   1 3993    6    0    0]
 [   0    1 3987   12    0]
 [   0    0    2 1705    6]
 [   0    0    0    1  242]]
```

### Classification Performance

| Grade | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| A     |      1.00 |   1.00 |     1.00 |
| B     |      1.00 |   1.00 |     1.00 |
| C     |      1.00 |   1.00 |     1.00 |
| D     |      0.99 |   1.00 |     0.99 |
| F     |      0.98 |   1.00 |     0.99 |

---

##  Visualization

A **confusion matrix heatmap** is generated using Seaborn.

The visualization compares:

* Actual Grade
* Predicted Grade

This makes it easier to identify which grades were correctly and incorrectly classified.

---

##  Model Architecture

```text
Input Layer
    │
    ├── Weekly Self Study Hours
    ├── Attendance Percentage
    ├── Class Participation
    └── Total Score
          │
          ▼
   Dense Layer (64)
       ReLU
          │
          ▼
   Dense Layer (32)
       ReLU
          │
          ▼
     Output Layer
      Grade A-F
```

---

##  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project

Open the notebook:

```text
DL_2.ipynb
```

### 3. Place the dataset

Make sure the following file is in the same directory:

```text
student_performance.csv
```

### 4. Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 5. Run the notebook

Open `DL_2.ipynb` in:

* Google Colab
* Jupyter Notebook
* JupyterLab

and execute the cells.

---

##  Project Structure

```text
Student-Performance-Prediction/
│
├── DL_2.ipynb
├── student_performance.csv
└── README.md
```

---

##  Key Findings

* Student grades can be classified using academic and participation-related features.
* Feature scaling improves the suitability of numerical features for the MLP model.
* The MLP model achieved **99.73% test accuracy** on the sampled dataset.
* The confusion matrix shows very few incorrect predictions.
* Grades A, B, and C were classified with approximately perfect precision, recall, and F1-score in the reported results.

---

##  Conclusion

This project demonstrates how an **MLP neural network** can be used for multi-class student grade prediction.

By using study hours, attendance, class participation, and total score as input features, the model successfully classified students into grades **A, B, C, D, and F** with a reported accuracy of **99.73%**.

The project provides a practical example of applying neural-network-based classification to student performance data.

