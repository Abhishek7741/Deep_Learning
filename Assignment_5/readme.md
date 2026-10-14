# Assignment 05 – RNN vs LSTM vs GRU for Sequence Classification

## 📌 Aim

To implement and compare Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) models for sequence classification and analyze their performance using appropriate evaluation metrics.

## 📖 Problem Statement

Implement and compare RNN, LSTM, and GRU models for sequence classification, and evaluate their performance using accuracy, precision, recall, F1-score, training time, and confusion matrices.

## 📊 Dataset Description

This project uses the **Reuters Newswire Dataset**, available through TensorFlow/Keras.

The dataset contains news articles classified into **46 different topics**. Each article is represented as a sequence of integer word IDs, and the objective is to predict the topic of the article.

Example topic categories include:
- Economics
- Trade
- Politics
- Business
- Commodities
- Finance

### Dataset Information

| Property | Value |
|---|---|
| Dataset | Reuters Newswire |
| Training samples | 8,982 |
| Testing samples | 2,246 |
| Number of classes | 46 |
| Maximum vocabulary size | 10,000 |
| Maximum sequence length | 200 |

## 🧠 Models Used

### 1. Recurrent Neural Network (RNN)

RNNs are designed to process sequential data. They maintain a hidden state that carries information from previous time steps. However, standard RNNs can suffer from the vanishing-gradient problem, making long-term dependencies difficult to learn.

### 2. Long Short-Term Memory (LSTM)

LSTM is an improved RNN architecture that uses memory cells and gates to retain important information over longer periods. Its main gates are the forget gate, input gate, and output gate.

### 3. Gated Recurrent Unit (GRU)

GRU is a gated recurrent architecture that uses update and reset gates to control information flow. It can learn long-term dependencies with fewer parameters than LSTM in many configurations.

## ⚙️ Data Preprocessing

The following steps are performed:

1. Import the required Python libraries.
2. Load the Reuters dataset using TensorFlow/Keras.
3. Limit the vocabulary to the 10,000 most frequent word IDs.
4. Examine the distribution of article sequence lengths.
5. Pad or truncate sequences to a fixed length of 200.
6. Use an embedding layer to represent words as dense vectors.
7. Split the training data further into training and validation sets during model fitting.
8. Train and evaluate the RNN, LSTM, and GRU models on the same dataset.

## 🏗️ Model Architecture

All three models use a common architecture, with the recurrent layer changed for each experiment.

```text
Input: Padded News Article
          ↓
Embedding Layer
Vocabulary Size: 10,000
Embedding Dimension: 128
          ↓
Recurrent Layer
(SimpleRNN / LSTM / GRU)
          ↓
Dropout – 0.3
          ↓
Dense Layer – 64 Neurons
ReLU Activation
          ↓
Dropout – 0.2
          ↓
Dense Layer – 46 Neurons
Softmax Activation
          ↓
Predicted News Category
```

### Model Configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Maximum sequence length | 200 |
| Embedding dimension | 128 |
| Recurrent hidden units | 128 |
| Dense layer | 64 neurons |
| First dropout rate | 0.3 |
| Second dropout rate | 0.2 |
| Output classes | 46 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse Categorical Crossentropy |
| Batch size | 128 |
| Maximum epochs | 5 |
| Validation split | 20% |
| Early stopping patience | 2 |

Early stopping monitors validation loss and restores the best weights when training stops.

## 🚀 Methodology

1. Load the Reuters Newswire dataset.
2. Set random seeds for reproducibility.
3. Analyze article sequence lengths and topic distribution.
4. Pad the input sequences to 200 tokens.
5. Create an embedding layer.
6. Build the RNN model and train it.
7. Build the LSTM model and train it.
8. Build the GRU model and train it.
9. Generate predictions for the test dataset.
10. Calculate accuracy, precision, recall, and F1-score.
11. Generate classification reports and confusion matrices.
12. Record model training times and parameter counts.
13. Compare performance using tables and visualizations.
14. Identify the best-performing model for each metric.

## 📏 Evaluation Metrics

- **Accuracy:** Percentage of correctly classified news articles.
- **Precision:** Proportion of predicted positive classifications that are correct, calculated using weighted averaging.
- **Recall:** Proportion of actual classifications correctly identified, calculated using weighted averaging.
- **F1-score:** Harmonic mean of precision and recall, calculated using weighted averaging.
- **Confusion Matrix:** Shows correct and incorrect classifications across the 46 classes.
- **Training Time:** Measures the time required to train each model.
- **Parameter Count:** Measures the number of model parameters.

## 📈 Experimental Results

The following results were obtained from the notebook execution.

| Metric | RNN | LSTM | GRU |
|---|---:|---:|---:|
| Accuracy | 44.52% | 57.26% | 56.77% |
| Precision | 41.26% | 51.78% | 52.15% |
| Recall | 44.52% | 57.26% | 56.77% |
| F1-score | 39.93% | 53.23% | 52.80% |
| Training time | 128.94 sec | 348.47 sec | 243.84 sec |
| Parameters | 1,324,142 | 1,422,830 | 1,390,318 |

*Note: Training time depends on the hardware and runtime environment.*

## 🏆 Best Model by Metric

Based on the recorded experimental results:

- **Best accuracy:** LSTM — 57.26%
- **Best precision:** GRU — 52.15%
- **Best recall:** LSTM — 57.26%
- **Best F1-score:** LSTM — 53.23%
- **Lowest training time:** RNN — 128.94 seconds

LSTM achieved the highest accuracy, recall, and F1-score in this experiment. GRU achieved the highest weighted precision, while the standard RNN had the shortest training time.

## 📉 Visualizations

The notebook generates the following visualizations:

1. **Sequence Length Distribution:** Shows the distribution of news article lengths and the maximum sequence length of 200.
2. **Class Distribution:** Displays the number of training articles in each of the 46 topic classes.
3. **Training Accuracy Comparison:** Compares training accuracy across RNN, LSTM, and GRU.
4. **Validation Accuracy Comparison:** Compares validation accuracy across the three architectures.
5. **Training Loss Comparison:** Shows how training loss changes over epochs.
6. **Validation Loss Comparison:** Compares validation loss during training.
7. **Metric Comparison:** Compares accuracy, precision, recall, and F1-score.
8. **Training Time Comparison:** Compares
