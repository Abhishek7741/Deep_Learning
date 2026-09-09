# Iris Classification Using Neural Network

##  Project Overview

This project implements a **Neural Network using TensorFlow/Keras** to classify Iris flowers into three different species.

The project also studies how changing the **learning rate** and **number of epochs** affects the performance of the neural network.

The Iris dataset is loaded using Scikit-learn's `load_iris()` dataset.

---

##  Objective

The main objectives of this project are:

* To build a neural network for Iris flower classification.
* To train the model using different learning rates.
* To train the model for different numbers of epochs.
* To compare the test accuracy for each combination.
* To understand the effect of learning rate and training duration on model performance.

---

##  Dataset

The project uses the **Iris dataset**.

The dataset contains four input features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The model classifies each flower into one of three classes:

* Setosa
* Versicolor
* Virginica

The target labels are converted into one-hot encoded vectors using `to_categorical()`.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* NumPy

---

##  Libraries Used

```python
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.utils import to_categorical
```

---

##  Methodology

### 1. Load the Dataset

The Iris dataset is loaded using:

```python
iris = load_iris()
X = iris.data
y = to_categorical(iris.target)
```

The four numerical features are used as input to the neural network.

---

### 2. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 3. Feature Scaling

`StandardScaler` is used to standardize the input features.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Scaling helps the neural network train more effectively.

---

##  Neural Network Architecture

The model uses the following architecture:

```text
Input Layer
4 Features
    │
    ▼
Dense Layer
16 Neurons
ReLU Activation
    │
    ▼
Dense Layer
8 Neurons
ReLU Activation
    │
    ▼
Output Layer
3 Neurons
Softmax Activation
```

### Model Configuration

| Component         | Configuration            |
| ----------------- | ------------------------ |
| Input features    | 4                        |
| Hidden Layer 1    | 16 neurons               |
| Hidden Layer 2    | 8 neurons                |
| Output Layer      | 3 neurons                |
| Hidden activation | ReLU                     |
| Output activation | Softmax                  |
| Optimizer         | Adam                     |
| Loss function     | Categorical Crossentropy |
| Batch size        | 16                       |

---

##  Experiment

The project compares three different learning rates:

```text
0.001
0.01
0.1
```

and three different numbers of epochs:

```text
20
50
100
```

Therefore, a total of **9 different combinations** are tested.

For every combination, a new neural network is created, trained, and evaluated on the test dataset.

---

##  Results

The obtained test accuracies are:

| Learning Rate |   20 Epochs |   50 Epochs |  100 Epochs |
| ------------: | ----------: | ----------: | ----------: |
|     **0.001** |      83.33% |      90.00% | **100.00%** |
|      **0.01** |      96.67% | **100.00%** | **100.00%** |
|       **0.1** | **100.00%** | **100.00%** |      96.67% |

---

##  Best Performance

The experiment achieved a maximum test accuracy of:

### **100%**

This occurred for multiple configurations:

* Learning rate **0.001**, 100 epochs
* Learning rate **0.01**, 50 epochs
* Learning rate **0.01**, 100 epochs
* Learning rate **0.1**, 20 epochs
* Learning rate **0.1**, 50 epochs

---

##  Observations

### Learning Rate = 0.001

The model starts with lower accuracy at 20 epochs:

```text
20 epochs  → 83.33%
50 epochs  → 90.00%
100 epochs → 100.00%
```

This shows that the smaller learning rate requires more training epochs to reach high accuracy.

### Learning Rate = 0.01

The model performs very well:

```text
20 epochs  → 96.67%
50 epochs  → 100.00%
100 epochs → 100.00%
```

This provides consistently strong performance.

### Learning Rate = 0.1

The model reaches 100% accuracy quickly:

```text
20 epochs  → 100.00%
50 epochs  → 100.00%
100 epochs → 96.67%
```

The decrease at 100 epochs indicates that simply increasing the number of epochs does not always improve test performance.

---

##  Key Learning

This experiment demonstrates that **learning rate and number of epochs have a significant effect on neural network training**.

A very small learning rate may require more epochs, while a larger learning rate can learn faster but may not always produce the best result when training continues for too long.

Therefore, selecting suitable hyperparameters is important for obtaining good model performance.

---

##  How to Run

### 1. Install Dependencies

```bash
pip install tensorflow scikit-learn
```

### 2. Open the Notebook

Open:

```text
DL_3.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 3. Run All Cells

Execute the notebook to train the neural network and generate the accuracy comparison table.

---

##  Project Structure

```text
Iris-Neural-Network/
│
├── DL_3.ipynb
└── README.md
```

---

##  Conclusion

The project successfully demonstrates **multi-class Iris flower classification using a neural network**.

By experimenting with different learning rates and epochs, the results show that the model can achieve **100% test accuracy** with several hyperparameter combinations.

The experiment also demonstrates that increasing epochs does not necessarily guarantee better performance, as shown by the decrease from **100% to 96.67%** for the learning rate of `0.1` when training was increased from 50 to 100 epochs.

