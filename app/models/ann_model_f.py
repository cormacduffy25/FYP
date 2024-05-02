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

    joblib.dump(sc, 'scaler.gz')

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

    model.save('ann_model_f.h5')

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

def get_initial_features():
    sql_query = """
    SELECT
        coalprice as coalprice_lag1, oilprice as oilprice_lag1, gasprice as gasprice_lag1, nuclearprice as nuclearprice_lag1, hydroprice as hydroprice_lag1, windsolarprice as windsolarprice_lag1, cokebreezeprice as cokebreezeprice_lag1,
        LAG(coalprice, 1) OVER (ORDER BY year) as coalprice_lag2, LAG(oilprice, 1) OVER (ORDER BY year) as oilprice_lag2, LAG(gasprice, 1) OVER (ORDER BY year) as gasprice_lag2, LAG(nuclearprice, 1) OVER (ORDER BY year) as nuclearprice_lag2, LAG(hydroprice, 1) OVER (ORDER BY year) as hydroprice_lag2, LAG(windsolarprice, 1) OVER (ORDER BY year) as windsolarprice_lag2, LAG(cokebreezeprice, 1) OVER (ORDER BY year) as cokebreezeprice_lag2,
        LAG(coalprice, 2) OVER (ORDER BY year) as coalprice_lag3, LAG(oilprice, 2) OVER (ORDER BY year) as oilprice_lag3, LAG(gasprice, 2) OVER (ORDER BY year) as gasprice_lag3, LAG(nuclearprice, 2) OVER (ORDER BY year) as nuclearprice_lag3, LAG(hydroprice, 2) OVER (ORDER BY year) as hydroprice_lag3, LAG(windsolarprice, 2) OVER (ORDER BY year) as windsolarprice_lag3, LAG(cokebreezeprice, 2) OVER (ORDER BY year) as cokebreezeprice_lag3
    FROM fuel_sources_lagged
    WHERE year = 2022
    """
    engine = create_engine(db_url)
    data = pd.read_sql_query(sql_query, engine)
    if not data.empty:
        return data.iloc[0].values.reshape(1, -1)  # Ensure the output is in 2D array format
    else:
        raise Exception("No data found for prediction")
    
def update_lagged_features(features, new_predictions):
    new_features = np.roll(features, -len(new_predictions))
    new_features[-len(new_predictions):] = new_predictions
    return new_features

def recursive_forecasting(model, initial_features, scaler, steps=3):
    predictions = []  # This will store all predictions
    current_features = initial_features

    for _ in range(steps):
        current_features_scaled = scaler.transform(current_features)
        prediction = model.predict(current_features_scaled)[0]  # Single prediction
        predictions.append(prediction)  # Append single prediction to the list
        current_features = update_lagged_features(current_features.flatten(), prediction).reshape(1, -1)

    return predictions

def main():
    model_path = 'ann_model_f.h5'
    scaler_path = 'scaler.gz'
    steps = 3

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    initial_features = get_initial_features()

    predictions = recursive_forecasting(model, initial_features, scaler, steps=steps)

    for i, prediction in enumerate(predictions):
        print(f"Forecast for Step {i+1}: {prediction}")

if __name__ == "__main__":
    main()
