# Assignment 08 – Sentiment Analysis Using Pretrained BERT

## 📌 Aim

To implement a pretrained BERT model for sentiment analysis and text classification using a sample movie-review dataset.

## 📝 Problem Statement

Implement a pretrained BERT model for sentiment analysis or text classification on a sample dataset and evaluate its performance using appropriate evaluation metrics.

## 📖 Introduction

Sentiment analysis is a Natural Language Processing (NLP) task used to identify the emotional tone or opinion expressed in text.

It is widely used to analyze movie reviews, product reviews, customer feedback, social media posts, and survey responses.

This assignment uses **BERT (Bidirectional Encoder Representations from Transformers)**, a pretrained Transformer-based language model, to classify movie reviews as positive or negative.

## 📊 Dataset Description

The project uses the Rotten Tomatoes movie-review dataset, which contains short movie-review sentences labeled according to their sentiment.

### Sentiment Classes

| Label | Sentiment |
|---:|---|
| 0 | Negative |
| 1 | Positive |

### Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset | Rotten Tomatoes |
| Training samples | 2,000 |
| Testing samples | 500 |
| Number of classes | 2 |
| Training epochs | 1 |
| Random seed | 42 |
| Maximum sequence length | 256 tokens |

The dataset is downloaded automatically using the Hugging Face Datasets library. A smaller subset is selected to reduce computational requirements.

## 🧠 What Is BERT?

BERT stands for **Bidirectional Encoder Representations from Transformers**.

It is a Transformer-based language model that learns contextual relationships between words by considering text bidirectionally.

### Key Features

- Transformer-based architecture.
- Bidirectional contextual understanding.
- Pretrained on large amounts of text.
- Can be fine-tuned for specific NLP tasks.
- The uncased version does not distinguish between uppercase and lowercase letters.

## ⚙️ Model Used

The project uses the pretrained **`bert-base-uncased`** model.

The pretrained BERT model is loaded using Hugging Face Transformers. A sequence-classification head is added for the two sentiment classes.

The model is fine-tuned on the selected Rotten Tomatoes training samples to learn sentiment classification.

## 🔄 Methodology

```text
Rotten Tomatoes Dataset
          ↓
Select Training and Testing Samples
          ↓
Explore Dataset
          ↓
Load BERT Tokenizer
          ↓
Tokenization and Padding
          ↓
Load Pretrained BERT
          ↓
Add Classification Layer
          ↓
Fine-Tune Model
          ↓
Evaluate Performance
          ↓
Accuracy, Precision, Recall and F1-score
          ↓
Confusion Matrix
          ↓
Predict Sentiment of New Reviews
          ↓
Save and Reload Trained Model
```

## 🛠️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the Rotten Tomatoes dataset.
2. Select 2,000 training samples and 500 test samples.
3. Explore sample reviews and sentiment labels.
4. Analyze the distribution of positive and negative reviews.
5. Tokenize the text using the pretrained BERT tokenizer.
6. Truncate sequences to a maximum length of 256 tokens.
7. Dynamically pad batches using `DataCollatorWithPadding`.
8. Convert tokenized data into the format required by the model.

### Tokenization Example

Input sentence:

```text
This movie was absolutely fantastic!
```

Example tokens:

```text
['this', 'movie', 'was', 'absolutely', 'fantastic', '!']
```

BERT converts these tokens into numerical token IDs that the model can process.

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Model | bert-base-uncased |
| Optimizer | AdamW through Hugging Face Trainer |
| Learning rate | 0.00002 |
| Training epochs | 1 |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Weight decay | 0.01 |
| Loss function | Cross-entropy classification loss |
| Evaluation metric for best model | F1-score |
| Random seed | 42 |
| Maximum sequence length | 256 |

The Hugging Face `Trainer` API is used to simplify training, evaluation, and model saving.

## 📈 Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy:** Percentage of correctly classified movie reviews.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures the ability to identify actual positive samples.
- **F1-score:** Combines precision and recall.
- **Confusion Matrix:** Displays correct and incorrect predictions for both sentiment classes.

## 📊 Experimental Results

The following results were obtained during execution on the selected 500-review test subset.

| Metric | Result |
|---|---:|
| Test accuracy | 82.80% |
| Precision | 81.86% |
| Recall | 81.86% |
| F1-score | 81.86% |
| Test loss | 0.4149 |
| Test samples | 500 |

### Class-wise Classification Report

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 83.65% | 83.65% | 83.65% | 263 |
| Positive | 81.86% | 81.86% | 81.86% | 237 |
| **Overall accuracy** | | | **82.80%** | **500** |

## 🔍 Confusion Matrix

The confusion matrix obtained during evaluation is:

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 220 | 43 |
| Positive | 43 | 194 |

Interpretation:

- 220 negative reviews were correctly classified.
- 194 positive reviews were correctly classified.
- 43 negative reviews were incorrectly classified as positive.
- 43 positive reviews were incorrectly classified as negative.

The model correctly classified 414 out of 500 reviews.

## 🧪 Testing on New Sentences

The fine-tuned model is also tested on new movie-review sentences.

| Review | Predicted sentiment | Confidence |
|---|---|---:|
| This movie was absolutely fantastic. I loved every minute of it! | Positive | 94.18% |
| The movie was boring and disappointing. | Negative | 91.78% |
| The acting was excellent and the story was very entertaining. | Positive | 94.53% |
| I did not enjoy this movie at all. | Negative | 89.55% |
| A wonderful film with great performances. | Positive | 94.80% |
| The plot was terrible and the movie was a complete waste of time. | Negative | 89.80% |

Confidence represents the model's predicted probability for the selected sentiment class.

## 💾 Saving and Loading the Model

After fine-tuning, the model and tokenizer are saved locally.

```python
SAVE_PATH = "./bert_rotten_tomatoes_model"

trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
```

The saved model can be loaded again for future sentiment predictions without repeating the complete training process.

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

## 📦 Installation

Install the required libraries:

```bash
pip install -U transformers datasets accelerate scikit-learn matplotlib seaborn torch numpy pandas
```

## ▶️ How to Run

1. Open Google Colab or Jupyter Notebook.
2. Open the Assignment 8 notebook
