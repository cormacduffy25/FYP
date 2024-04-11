import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from database.database import load_data_from_db

def train_ann_model():
    """
    Trains an Artificial Neural Network (ANN) model using the given dataset.
    """

    # Loading the Data
    train_data = load_data_from_db()
    
    features = ['Year', 'coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']
    X = train_data[features]  # Your input features
    y = train_data['avgprice']  # Your target variable

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialising the ANN 
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=128)
    # Evaluating the ANN
    model.evaluate(X_test, y_test)
    # Predicting the Test set results
    predictions = model.predict(X_test)
    # Flatten the predictions and actual values to 1D array
    predictions = predictions.flatten()
    actual_values = y_test.to_numpy()
    # Print the test loss and test MAE
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Print the first 10 predictions and actual values
    print("Prediction vs Actual")
    for i in range(len(predictions)):
        print(f"Prediction: {predictions[i]:.4f}, Actual: {actual_values[i]:.4f}")

    # Plot the predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label='Predictions', color='red', linestyle='--', marker='x')
    plt.plot(actual_values, label='Actual', color='blue', linestyle='--', marker='o')
    plt.title('Predictions vs Actual Values ANN Model')
    plt.legend()
    plt.show()
train_ann_model()