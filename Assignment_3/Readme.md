# Practical 3 – Forward Propagation & Backpropagation (Learning Rate / Epoch Analysis)

**Subject:** Deep Learning  
**Department:** CSE-AI  
**Semester:** 5  
**Academic Year:** 2026-27  
**Name:** Abhishek Kadam  
**Roll No:** 16  
**PRN:** 12412535  

---

## 📌 Problem Statement

Implement forward propagation and backpropagation using TensorFlow/Keras. Analyze the effect of different learning rates and the number of epochs on model performance.

---

## 📊 Dataset

This practical uses the **Fashion-MNIST dataset**.

Fashion-MNIST is a benchmark dataset for image classification. It contains **70,000 grayscale images** of fashion products belonging to **10 different classes**.

### Dataset Details

- Training Samples: 60,000
- Testing Samples: 10,000
- Image Size: 28 × 28 pixels
- Pixels per Image: 784
- Number of Classes: 10

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
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 🔄 Workflow

```text
Fashion-MNIST Dataset
        ↓
Data Loading
        ↓
Data Preprocessing
        ↓
Image Reshaping
        ↓
Normalization
        ↓
ANN Model Creation
        ↓
Forward Propagation
        ↓
Loss Calculation
        ↓
Backpropagation
        ↓
Adam Optimizer
        ↓
Learning Rate Analysis
        ↓
Epoch Analysis
        ↓
Performance Evaluation
        ↓
Visualization
🔧 Data Preprocessing

Each Fashion-MNIST image has a size of 28 × 28 pixels.

The images are flattened into a 784-dimensional vector and pixel values are normalized from 0–255 to 0–1.

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

During training, 20% of the training data is used as validation data.

🧠 ANN Model Architecture

The Artificial Neural Network used in this practical consists of two hidden layers.

Input Layer
784 Neurons
     ↓
Hidden Layer 1
128 Neurons
ReLU
     ↓
Hidden Layer 2
64 Neurons
ReLU
     ↓
Output Layer
10 Neurons
Softmax
Model Configuration
Parameter	Value
Input Layer	784 neurons
Hidden Layer 1	128 neurons
Hidden Layer 2	64 neurons
Output Layer	10 neurons
Hidden Activation	ReLU
Output Activation	Softmax
Optimizer	Adam
Loss Function	Sparse Categorical Crossentropy
Batch Size	128
🔁 Forward Propagation

Forward propagation passes the input data through the neural network from the input layer to the output layer.

Input Data
    ↓
Weights + Bias
    ↓
ReLU Activation
    ↓
Hidden Layers
    ↓
Softmax
    ↓
Output Probabilities

The output represents the predicted probability for each of the 10 Fashion-MNIST classes.

🔙 Backpropagation

Backpropagation updates the network weights based on the prediction error.

Prediction
    ↓
Calculate Loss
    ↓
Calculate Error
    ↓
Propagate Error Backward
    ↓
Calculate Gradients
    ↓
Update Weights
    ↓
Improve Model

The Adam optimizer is used to update the weights and minimize the loss.

📈 Learning Rate Analysis

Different learning rates are tested to study their effect on model performance.

Learning Rates Tested
0.1
0.01
0.001

The accuracy obtained for each learning rate is compared using a Learning Rate vs Accuracy graph.

⏱️ Epoch Analysis

The effect of the number of epochs on model performance is also analyzed.

Epochs Tested
5
10
20

The model is trained for each epoch value and the resulting accuracy is compared using an Epochs vs Accuracy graph.

📊 Performance Evaluation

The final model is evaluated using:

Accuracy
Confusion Matrix
Classification Report
Misclassified Images
Training Configuration
Learning Rate  : 0.001
Epochs         : 20
Batch Size     : 128
Validation     : 20%
📈 Visualizations

The practical generates the following visualizations:

Learning Rate vs Accuracy
Epochs vs Accuracy
Training vs Validation Accuracy
Training vs Validation Loss
Confusion Matrix
Classification Report
Misclassified Images

These visualizations help analyze model performance and identify classes that are difficult to classify.

🎨 Graph Customization

The graphs have been given small presentation-level customizations such as:

Customized graph colors
Different color maps
Bold graph titles
Adjusted figure sizes
Improved visualization formatting

These changes are only for presentation and do not change the ANN architecture, dataset, optimizer, loss function, or training procedure.

❌ Misclassified Images

A sample of incorrectly classified Fashion-MNIST images is visualized.

Each image shows:

True Label
Predicted Label

This helps identify visually similar clothing categories that the model may confuse.

▶️ How to Run
1. Open the Notebook

Open the notebook in Google Colab or Jupyter Notebook.

2. Install/Import Required Libraries

Run the required TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn imports.

3. Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
4. Preprocess the Dataset

Reshape the images into 784-dimensional vectors and normalize the pixel values.

5. Create the ANN Model

Use the architecture:

784 → 128 → 64 → 10
6. Analyze Learning Rates

Test the following learning rates:

0.1
0.01
0.001
7. Analyze Epochs

Test the following epoch values:

5
10
20
8. Train and Evaluate the Final Model

Train the model using the selected configuration and generate the accuracy, confusion matrix, classification report, and misclassified images.
