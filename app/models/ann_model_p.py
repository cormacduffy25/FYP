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
import keras_tuner as kt

# Callback to print training progress every 50 epochs
class PrintEvery50Epochs(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: loss - {logs['loss']:.4f}, mae - {logs['mae']:.4f}, val_loss - {logs['val_loss']:.4f}, val_mae - {logs['val_mae']:.4f}")

# Function to retrieve database URL from config file
def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

# Creating a database engine
db_url = get_db_url()
engine = create_engine(db_url)

# Function to load data from the database
def load_data_from_db():
    sql_query = 'SELECT * FROM fuelsources'
    return pd.read_sql_query(sql_query, engine)

# Function to save predictions to the database
def save_predictions_to_db(actuals, predictions, years, actuals_train, predictions_train, years_train):
    test_df = pd.DataFrame({
        'type': 'test',
        'actual': actuals,
        'prediction': predictions,
        'year': years,
    })
    train_df = pd.DataFrame({
        'type': 'train',
        'actual': actuals_train,
        'prediction': predictions_train,
        'year': years_train})

    combined_df = pd.concat([test_df, train_df])
    combined_df.to_sql('predictions_ann', con=engine, if_exists='replace', index=False)
    print("Data saved to database successfully.")

# Function to plot actual vs predicted values
def plot_act_vs_predicted(predicted, actual, title):
    plt.figure(figsize=(12, 6))
    plt.plot(predicted, label='Predictions', color='red', linestyle='--', marker='x')
    plt.plot(actual, label='Actual', color='blue', linestyle='--', marker='o')
    plt.title(f'Predictions vs Actual Values ANN Model ({title})')
    plt.legend()
    plt.show()
    
    
def train_ann_model():
    """
    Trains an Artificial Neural Network (ANN) model using the given dataset.
    """

    # Loading the Data 
    train_data = load_data_from_db()

    print(train_data.head())
    
    features = ['year', 'coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']
    x = train_data[features]  # Your input features
    y = train_data['avgprice']  # Your target variable

    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, shuffle=True)

    # Extracting the years from the dataset
    years_test = x_test['year'].values
    years_train = x_train['year'].values

    # Feature Scaling
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    # Initialising the ANN 
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    #Print Model Summary 
    print(model.summary())
    # Compiling the ANN
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.05,beta_1=0.9, beta_2=0.999, epsilon=1e-07,), loss='mean_squared_error', metrics=['mae'])
    # Fitting the ANN to the Training set
    history = model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=500, batch_size=128, callbacks=[PrintEvery50Epochs()], verbose=0)
    # Evaluating the ANN
    model.evaluate(x_test_scaled, y_test)
    # Predicting the Test set results
    predictions_test = model.predict(x_test_scaled)
    preditctions_train = model.predict(x_train_scaled)
    # Flatten the predictions and actual values to 1D array
    predictions_test = predictions_test.flatten()
    preditctions_train = preditctions_train.flatten()

    actual_values_test = y_test.to_numpy()
    actual_values_train = y_train.to_numpy()

    # Save the predictions to the database
    save_predictions_to_db(actual_values_test, predictions_test, years_test, actual_values_train, preditctions_train, years_train)

    epochs = history.epoch
    indices = [i for i in epochs if i % 50 == 0 or i == epochs[0]]

        # Plot training & validation loss values every 50 epochs
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

    plot_act_vs_predicted(predictions_test, actual_values_test, "Test Data")
    plot_act_vs_predicted(preditctions_train, actual_values_train, "Train Data")

       # Print the test predictions and actual values
    print("Prediction vs Actual")
    for i in range(len(predictions_test)):
        print(f"Prediction: {predictions_test[i]:.4f}, Actual: {actual_values_test[i]:.4f}")
    # Print the train predictions and actual values
    print("Prediction vs Actual")
    for i in range(len(preditctions_train)):
        print(f"Prediction: {preditctions_train[i]:.4f}, Actual: {actual_values_train[i]:.4f}")

     # Print the test loss and test MAE
    test_loss, test_mae = model.evaluate(x_test_scaled, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
train_ann_model()