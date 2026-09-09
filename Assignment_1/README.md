# Deep Learning Binary Classification

##  Project Overview

This project implements a **Deep Learning model for binary classification** using **TensorFlow and Keras**.

The dataset is loaded from `data.csv`, preprocessed using label encoding and standardization, and then divided into training and testing datasets. A **Feed-Forward Neural Network** is trained to perform binary classification.

The project also evaluates the trained model using accuracy, confusion matrix, classification report, and training/validation graphs.

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* TensorFlow
* Keras
* Scikit-learn
* Jupyter Notebook

---

##  Project Structure

```text
Deep-Learning/
│
├── DL__1.ipynb
├── data.csv
└── README.md
```

---

##  Dataset

The project uses a CSV dataset named:

```
data.csv
```

The target column used for classification is:

```
diagnosis
```

The target variable is converted into numerical values using `LabelEncoder`.

The remaining columns are used as input features.

---

##  Workflow

The project follows these main steps:

### 1. Import Libraries

Required Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, TensorFlow, and Scikit-learn are imported.

### 2. Load Dataset

The dataset is loaded using:

```python
df = pd.read_csv("data.csv")
```

### 3. Explore the Dataset

The notebook checks:

* Dataset shape
* Data types
* Statistical information
* Missing values

### 4. Encode Target Variable

The `diagnosis` column is converted into numerical values using `LabelEncoder`.

```python
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])
```

### 5. Separate Features and Target

The input features are stored in `X` and the target variable is stored in `y`.

```python
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
```

### 6. Feature Scaling

The features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### 7. Train-Test Split

The dataset is divided into:

* **80% Training data**
* **20% Testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 8. Build Neural Network

A Sequential neural network is created with the following architecture:

```text
Input Layer
     ↓
Dense Layer - 64 neurons, ReLU
     ↓
Dense Layer - 32 neurons, ReLU
     ↓
Dense Layer - 16 neurons, ReLU
     ↓
Output Layer - 1 neuron, Sigmoid
```

The model contains **30 input features**.

### 9. Compile the Model

The model uses:

* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Metric:** Accuracy

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### 10. Train the Model

The model is trained for:

* **50 epochs**
* **Batch size:** 16
* **Validation split:** 20%

```python
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)
```

---

## 📈 Model Evaluation

After training, the model is evaluated on the test dataset.

The following evaluation techniques are used:

### Test Accuracy

The model's performance is measured using test loss and accuracy.

### Confusion Matrix

A confusion matrix is generated to visualize:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

### Accuracy Graph

The training and validation accuracy are plotted against the number of epochs.

### Loss Graph

The training and validation loss are plotted against the number of epochs.

---

##  How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd Deep-Learning
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open:

```text
DL__1.ipynb
```

Make sure `data.csv` is present in the same directory as the notebook.

### 6. Run All Cells

Run the notebook cells sequentially to:

1. Load the dataset
2. Preprocess the data
3. Build the neural network
4. Train the model
5. Evaluate the model
6. Display the confusion matrix
7. Display the classification report
8. Display accuracy and loss graphs

---

##  Model Configuration

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Model             | Sequential Neural Network |
| Input Features    | 30                        |
| Hidden Layers     | 3                         |
| Hidden Neurons    | 64, 32, 16                |
| Hidden Activation | ReLU                      |
| Output Neurons    | 1                         |
| Output Activation | Sigmoid                   |
| Optimizer         | Adam                      |
| Loss Function     | Binary Crossentropy       |
| Epochs            | 50                        |
| Batch Size        | 16                        |
| Test Size         | 20%                       |
| Validation Split  | 20%                       |

---

##  Results

The notebook generates the following results after training:

* Test Loss
* Test Accuracy
* Confusion Matrix
* Classification Report
* Training vs Validation Accuracy graph
* Training vs Validation Loss graph

The exact numerical results depend on the dataset and execution environment.

---

##  Learning Objectives

This project demonstrates how to:

* Load and explore a dataset
* Handle categorical target variables
* Perform feature scaling
* Split data into training and testing sets
* Build a neural network using Keras
* Train a binary classification model
* Evaluate model performance
* Interpret a confusion matrix
* Generate a classification report
* Visualize training and validation performance

---

##  Author

**Abhishek Kadam**

---

##  License

This project is created for **educational and academic purposes**.
