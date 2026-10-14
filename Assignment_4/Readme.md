# Assignment 04 – LSTM-Based Time-Series Forecasting

## 📌 Aim

To develop a **Long Short-Term Memory (LSTM) neural network** for time-series forecasting using the Jena Climate Dataset. The model uses historical weather observations to predict the temperature for the next hour.

## 📊 Dataset Description

The project uses the **Jena Climate Dataset**, containing weather observations recorded at the Max Planck Institute for Biogeochemistry weather station in Jena, Germany.

- **Dataset period:** 2009–2016
- **Original observation interval:** 10 minutes
- **Processed observation interval:** 1 hour
- **Original dataset size:** 420,551 observations
- **Selected features:** 7
- **Target variable:** Temperature (`T (degC)`)

### Features Used

| Feature | Description |
|---|---|
| `p (mbar)` | Atmospheric pressure |
| `T (degC)` | Air temperature |
| `Tpot (K)` | Potential temperature |
| `Tdew (degC)` | Dew-point temperature |
| `rh (%)` | Relative humidity |
| `wv (m/s)` | Wind velocity |
| `max. wv (m/s)` | Maximum wind velocity |

## 🧠 What Is LSTM?

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to learn temporal relationships in sequential data.

An LSTM uses memory cells and gates to retain important information from previous time steps. This makes it useful for forecasting tasks such as weather prediction, stock-price forecasting, and sales prediction.

The main components of an LSTM cell are:

- **Forget Gate:** Determines which previous information should be discarded.
- **Input Gate:** Controls which new information should be stored.
- **Cell State:** Maintains information over time.
- **Output Gate:** Determines the information passed to the next step.

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Download the Jena Climate Dataset.
2. Load the CSV file using Pandas.
3. Convert the `Date Time` column into datetime format.
4. Set the datetime column as the index.
5. Select the seven relevant weather features.
6. Handle missing values using interpolation and remove remaining missing values.
7. Resample observations into hourly averages.
8. Split the dataset chronologically into training and testing sets.
9. Normalize the features using `MinMaxScaler`.
10. Create sequences using the previous 24 hourly observations.

### Train-Test Split

- **Training data:** 80%
- **Testing data:** 20%

The split is chronological to preserve the time-series order.

## ⏱️ Sequence Creation

The model uses the previous **24 hours of weather observations** to predict the temperature at the next time step.

Each input sequence contains 24 time steps and 7 weather features.

```text
Previous 24 Hours of Weather Data
              ↓
          LSTM Model
              ↓
     Next-Hour Temperature
```

## 🏗️ LSTM Model Architecture

The model is built using TensorFlow/Keras.

```text
Input: 24 Time Steps × 7 Features
                 ↓
          LSTM – 64 Units
        return_sequences=True
                 ↓
           Dropout – 0.2
                 ↓
           LSTM – 32 Units
                 ↓
           Dropout – 0.2
                 ↓
       Dense – 16 Neurons
          ReLU Activation
                 ↓
         Dense – 1 Neuron
                 ↓
    Predicted Temperature (°C)
```

### Model Configuration

| Parameter | Value |
|---|---|
| First LSTM layer | 64 units |
| Second LSTM layer | 32 units |
| Dense hidden layer | 16 neurons |
| Dropout rate | 0.2 |
| Output neurons | 1 |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Additional training metric | Mean Absolute Error (MAE) |
| Batch size | 64 |
| Maximum epochs | 30 |
| Validation split | 10% |
| Lookback window | 24 hours |
| Early stopping patience | 5 |

Early stopping monitors validation loss and restores the best model weights. Training stops early if validation loss does not improve for five consecutive epochs.

## 🚀 Methodology

1. Import the required Python libraries.
2. Download and extract the Jena Climate Dataset.
3. Read the dataset and process timestamps.
4. Select the seven weather features.
5. Handle missing values and resample the data hourly.
6. Visualize historical temperature.
7. Split the data into training and testing sets.
8. Apply Min-Max normalization.
9. Generate 24-hour input sequences.
10. Build and compile the LSTM model.
11. Train the model with early stopping.
12. Predict temperatures for the test sequences.
13. Convert predictions back to the original temperature scale.
14. Evaluate the model using regression metrics.
15. Plot actual versus predicted temperatures.
16. Plot training and validation loss.
17. Predict the temperature for the next hour.

## 📈 Evaluation Metrics

The model is evaluated using four regression metrics:

- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the original temperature unit.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **R² Score:** Measures how well the predictions explain the variation in actual temperatures.

## 📊 Results

The following results were obtained in the practical execution:

| Metric | Result |
|---|---:|
| MSE | 11.6135 |
| RMSE | 3.4079 °C |
| MAE | 3.1376 °C |
| R² Score | 0.8228 |

The R² score of **0.8228** indicates that the model explains approximately 82.28% of the variance in the test temperatures.

### Next-Hour Prediction

The notebook produced the following next-hour temperature prediction:

```text
Predicted Temperature: -6.71 °C
```

This value corresponds to the final input sequence used in that notebook execution.

## 📉 Visualizations

### 1. Hourly Temperature

Displays temperature variations throughout the Jena Climate Dataset period.

### 2. Training and Validation Loss

Shows how the training and validation MSE change over the training epochs.

### 3. Actual vs Predicted Temperature

Compares the actual temperature with the LSTM-predicted temperature for the first 500 test time steps.

These plots help assess how closely the model follows temperature trends and how its prediction errors change during training.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## 📦 Installation

Install the required libraries:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## ▶️ How to Run

1. Clone or download the repository.
2. Open the project directory.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the notebook:

```text
Assignment_4_JENA_CLIMATE_DATASET.ipynb
```

5. Run all cells sequentially.

The dataset is downloaded automatically from the TensorFlow dataset repository during execution.

## 📁 Project Structure

```text
Assignment-4/
│
├── Assignment_4_JENA_CLIMATE_DATASET.ipynb
└── README.md
```

The dataset archive and extracted CSV are generated during execution and do not need to be committed to the repository.

## ✅ Conclusion

This assignment demonstrates how an LSTM neural network can be used for multivariate time-series forecasting. The model learns temporal patterns from 24 hours of weather observations and predicts the temperature at the next time step.

The experiment includes data preprocessing, sequence generation, LSTM model training, evaluation using MSE, RMSE, MAE, and R², and visualization of actual versus predicted temperatures.

The results demonstrate the potential of LSTM networks for weather forecasting and other sequential prediction tasks.

---

**Author:** Abhishek Kadam  
**Subject:** Deep Learning  
**Practical No.:** 4
