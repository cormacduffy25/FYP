import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, insert, MetaData, Table
import json
import os
import joblib
import kerastuner as kt

# Custom callback to print training progress every 50 epochs
class PrintEvery50Epochs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0 or epoch == 0:
           print(f"Epoch {epoch + 1}: loss - {logs['loss']:.4f}, mae - {logs['mae']:.4f}, val_loss - {logs['val_loss']:.4f}, val_mae - {logs['val_mae']:.4f}")

# Function to plot actual vs predicted values
def plot_act_vs_predicted(predicted, actual, title):
    plt.figure(figsize=(12, 6))
    plt.plot(predicted, label='Predictions', color='red', linestyle='--', marker='x')
    plt.plot(actual, label='Actual', color='blue', linestyle='--', marker='o')
    plt.title(f'Predictions vs Actual Values ANN Model ({title})')
    plt.legend()
    plt.show()

# Function to retrieve database URL from configuration file
def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

# Get the database URL
db_url = get_db_url()
# Create an engine to connect to the database
engine = create_engine(db_url)

# Function to load data from the database
def load_data_from_db():
   sql_query = 'SELECT * FROM fuel_sources_lagged'
   data = pd.read_sql_query(sql_query, engine)
   return data

# Function to save model metrics to a CSV file
def save_metrics_to_csv(test_loss, test_mae, filename='model_performance.csv'):
    # Create a DataFrame for the new row of metrics
    new_row = pd.DataFrame({'Test Loss': [test_loss], 'Test MAE': [test_mae]})
    
    # Check if the file exists
    if os.path.exists(filename):
        # If the file exists, load the existing data
        df = pd.read_csv(filename)
        # Concatenate the existing DataFrame with the new row
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        # If the file does not exist, use new_row as the initial DataFrame
        df = new_row

    # Save the updated DataFrame back to CSV
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

# Function to train the ANN model
def train_ann_model():
    # Load data from the database
    train_data = load_data_from_db()
    print(train_data.head())

    # Define feature and target columns
    feature_columns = [f'{fuel}_lag1' for fuel in ['coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']]
    feature_columns += [f'{fuel}_lag2' for fuel in ['coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']]
    feature_columns += [f'{fuel}_lag3' for fuel in ['coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']]
    target_columns = ['coalprice', 'oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']

    # Split features and target
    x = train_data[feature_columns]  # Input features
    y = train_data[target_columns]  # Target variable

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True, random_state=42)

    # Feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Save the scaler
    joblib.dump(sc, 'scaler.gz')

    # Define the model architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(y_train.shape[1], activation='relu')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.005), loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.1, callbacks=[PrintEvery50Epochs()], verbose=0)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(x_test, y_test)

    # Make predictions on the test set
    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_train)

    # Save the trained model
    model.save('ann_model_f.h5')

    # Get actual values from the test set
    actual_values_test = y_test.to_numpy()
    actual_values_train = y_train.to_numpy()

    # Extract coal, oil, and gas prices for plotting
    coal_index = target_columns.index('coalprice')
    actual_coal_prices = y_test['coalprice'].values
    predicted_coal_prices = predictions_test[:, coal_index]

    oil_index = target_columns.index('oilprice')
    actual_oil_prices = y_test['oilprice'].values
    predicted_oil_prices = predictions_test[:, oil_index]
    
    gas_index = target_columns.index('gasprice')
    actual_gas_prices = y_test['gasprice'].values
    predicted_gas_prices = predictions_test[:, gas_index]

    # Print predictions vs actuals for test data
    print("Prediction vs Actual - Test Data")
    for i in range(len(predictions_test)):  # Iterate over samples
        print(f"Sample {i + 1}")
        for j, fuel_type in enumerate(target_columns):  # Iterate over each fuel type
             print(f"{fuel_type} - Prediction: {predictions_test[i, j]:.4f}, Actual: {actual_values_test[i, j]:.4f}")

    # Print predictions vs actuals for train data
    print("Prediction vs Actual - Train Data")
    for i in range(len(predictions_train)):  # Iterate over samples
        print(f"Sample {i + 1}")
        for j, fuel_type in enumerate(target_columns):  # Iterate over each fuel type
            print(f"{fuel_type} - Prediction: {predictions_train[i, j]:.4f}, Actual: {actual_values_train[i, j]:.4f}")

    # Plot predictions vs actuals for train data
    plot_act_vs_predicted(predictions_train, actual_values_train, 'Train Data')
    # Plot predictions vs actuals for coal prices
    plot_act_vs_predicted(predicted_coal_prices, actual_coal_prices, 'Coal Price')
    # Plot predictions vs actuals for oil prices
    plot_act_vs_predicted(predicted_oil_prices, actual_oil_prices, 'Oil Price')
    # Plot predictions vs actuals for gas prices
    plot_act_vs_predicted(predicted_gas_prices, actual_gas_prices, 'Gas Price')

    # Print test loss and MAE
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Save model metrics to CSV
    save_metrics_to_csv(test_loss, test_mae)

    # Plot training & validation loss values every 50 epochs
    epochs = history.epoch
    indices = [i for i in epochs if i % 50 == 0 or i == epochs[0]]
    plt.figure(figsize=(12, 6))
    plt.plot([history.history['loss'][i] for i in indices], label='Loss (training data)', marker='o')
    plt.plot([history.history['val_loss'][i] for i in indices], label='Loss (validation data)', marker='o')
    plt.title('Model MSE Loss Every 50 Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(len(indices)), labels=[str(i+1) for i in indices])  # Set x-ticks to show epoch numbers
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Plot training & validation MAE values every 50 epochs
    plt.figure(figsize=(12, 6))
    plt.plot([history.history['mae'][i] for i in indices], label='MAE (training data)', marker='o')
    plt.plot([history.history['val_mae'][i] for i in indices], label='MAE (validation data)', marker='o')
    plt.title('Model MAE Every 50 Epochs')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.xticks(range(len(indices)), labels=[str(i+1) for i in indices])  # Set x-ticks to show epoch numbers
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

# Train the ANN model
train_ann_model()

# Function to retrieve initial features for recursive forecasting
def get_initial_features():
    sql_query = """
    SELECT
    coalprice as coalprice_lag1,
    coalprice_lag1 as coalprice_lag2,
    coalprice_lag2 as coalprice_lag3,
    oilprice as oilprice_lag1,
    oilprice_lag1 as oilprice_lag2,
    oilprice_lag2 as oilprice_lag3,
    gasprice as gasprice_lag1,
    gasprice_lag1 as gasprice_lag2,
    gasprice_lag2 as gasprice_lag3,
    nuclearprice as nuclearprice_lag1,
    nuclearprice_lag1 as nuclearprice_lag2,
    nuclearprice_lag2 as nuclearprice_lag3,
    hydroprice as hydroprice_lag1,
    hydroprice_lag1 as hydroprice_lag2,
    hydroprice_lag2 as hydroprice_lag3,
    windsolarprice as windsolarprice_lag1,
    windsolarprice_lag1 as windsolarprice_lag2,
    windsolarprice_lag2 as windsolarprice_lag3,
    cokebreezeprice as cokebreezeprice_lag1,
    cokebreezeprice_lag1 as cokebreezeprice_lag2,
    cokebreezeprice_lag2 as cokebreezeprice_lag3
    FROM fuel_sources_lagged
    WHERE year = 2022

    """
    engine = create_engine(db_url)
    data = pd.read_sql_query(sql_query, engine)
    print(data)
    if not data.empty:
        return data.iloc[0].values.reshape(1, -1)  # Ensure the output is in 2D array format
    else:
        raise Exception("No data found for prediction")

# Function to update lagged features for recursive forecasting
def update_lagged_features(features, new_predictions):
    new_features = np.roll(features, -len(new_predictions))
    new_features[-len(new_predictions):] = new_predictions
    return new_features

# Function for recursive forecasting
def recursive_forecasting(model, initial_features, scaler, steps=3):
    predictions = []  # This will store all predictions
    current_features = initial_features

    for _ in range(steps):
        current_features_scaled = scaler.transform(current_features)
        prediction = model.predict(current_features_scaled)[0]  # Single prediction
        predictions.append(prediction)  # Append single prediction to the list
        current_features = update_lagged_features(current_features.flatten(), prediction).reshape(1, -1)

    return predictions

# Main function for recursive forecasting
def main():
    model_path = 'ann_model_f.h5'
    scaler_path = 'scaler.gz'
    steps = 3

    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Get initial features for forecasting
    initial_features = get_initial_features()

    # Perform recursive forecasting
    predictions = recursive_forecasting(model, initial_features, scaler, steps=steps)

    # Print the forecasted prices
    for i, prediction in enumerate(predictions):
        print(f"Forecast for Step {i+1}: {prediction}")

# Execute the main function
if __name__ == "__main__":
    main()
