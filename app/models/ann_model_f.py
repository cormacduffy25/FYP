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

class PrintEvery50Epochs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0 or epoch == 0:
           print(f"Epoch {epoch + 1}: loss - {logs['loss']:.4f}, mae - {logs['mae']:.4f}, val_loss - {logs['val_loss']:.4f}, val_mae - {logs['val_mae']:.4f}")

def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

db_url = get_db_url()
engine = create_engine(db_url)
#predictions_ann = Table('predictions_ann', MetaData(), autoload_with=engine)

def load_data_from_db():
   sql_query = 'SELECT * FROM fuel_sources_lagged'
   data = pd.read_sql_query(sql_query, engine)
   return data

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
""""
def save_predictions_to_db(actuals, predictions, years, actuals_train, predictions_train, years_train):
    df = pd.DataFrame({
        'actual': actuals, 
        'prediction': predictions, 
        'year': years,
        'actual': actuals_train,
        'prediction': predictions_train,
        'year': years_train})    
    df.to_sql('predictions_ann', con=engine, if_exists='replace', index=False)
"""
def train_ann_model():

    train_data = load_data_from_db()
    print(train_data.head())

    feature_columns = [f'{fuel}_lag1' for fuel in ['coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']]
    feature_columns += [f'{fuel}_lag2' for fuel in ['coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']]
    feature_columns += [f'{fuel}_lag3' for fuel in ['coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']]
    target_columns = ['coalprice', 'oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']

    x = train_data[feature_columns]  # Your input features
    y = train_data[target_columns]  # Your target variable

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 32)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])

    model.compile(optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.001,rho=0.9), loss='mean_squared_error', metrics=['mae'])
    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.1, callbacks=[PrintEvery50Epochs()], verbose =0)
    test_loss, test_mae = model.evaluate(x_test, y_test)

    predictions_test = model.predict(x_test)
    predictions_train = model.predict(x_train)

    

    actual_values_test = y_test.to_numpy()
    actual_values_train = y_train.to_numpy()

    coal_index = target_columns.index('coalprice')
    actual_coal_prices = y_test['coalprice'].values
    predicted_coal_prices = predictions_test[:, coal_index]

    print("Prediction vs Actual - Test Data")
    for i in range(len(predictions_test)):  # Iterate over samples
        print(f"Sample {i + 1}")
        for j, fuel_type in enumerate(target_columns):  # Iterate over each fuel type
             print(f"{fuel_type} - Prediction: {predictions_test[i, j]:.4f}, Actual: {actual_values_test[i, j]:.4f}")

    print("Prediction vs Actual - Train Data")
    for i in range(len(predictions_train)):  # Iterate over samples
        print(f"Sample {i + 1}")
        for j, fuel_type in enumerate(target_columns):  # Iterate over each fuel type
         print(f"{fuel_type} - Prediction: {predictions_train[i, j]:.4f}, Actual: {actual_values_train[i, j]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(actual_coal_prices, 'o-', label='Actual Prices', color='blue')
    plt.plot(predicted_coal_prices, 'x-', label='Predicted Prices', color='red')
    plt.title('Actual vs. Predicted Coal Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Coal Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
    save_metrics_to_csv(test_loss, test_mae)
    
train_ann_model()
