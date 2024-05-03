import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error 
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import json

def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

db_url = get_db_url()
engine = create_engine(db_url)

def load_data_from_db():
   sql_query = 'SELECT * FROM fuelsources'
   return pd.read_sql_query(sql_query, engine)

def get_forecasted_fuel_prices():
    sql_query = 'SELECT * FROM fuelsources_forecasted_svm'
    return pd.read_sql_query(sql_query, engine)

def save_predictions_to_db(actuals_test, predictions_test, years_test, actuals_train, predictions_train, years_train, forecasted_prices, forecasted_years):
    # Creating separate DataFrames for test, train, and forecast data
    test_df = pd.DataFrame({
        'type': 'test',
        'actual': actuals_test, 
        'prediction': predictions_test, 
        'year': years_test
    })
    
    train_df = pd.DataFrame({
        'type': 'train',
        'actual': actuals_train,
        'prediction': predictions_train,
        'year': years_train
    })
    
    forecast_df = pd.DataFrame({
        'type': 'forecast',
        'actual': 0,
        'prediction': forecasted_prices,
        'year': forecasted_years
    })
    
    # Combining all DataFrames
    combined_df = pd.concat([test_df, train_df, forecast_df])
    
    # Saving the combined DataFrame to SQL
    combined_df.to_sql('svm_prediction', con=engine, if_exists='replace', index=False)
    print("Data saved to database successfully.")

def save_forecast_to_db(years, predictions):
    df = pd.DataFrame({
        'year': years,
        'predicted_avg_price': predictions
    })
    df.to_sql('fuelsources_forecasted__predicted_svm', con=engine, if_exists='replace', index=False)
    print("Forecasted data saved to database successfully.")

def train_svm_model():

     # Loading the Data
    train_data = load_data_from_db()
    print(train_data.head())

    features = ['year', 'coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice']
    X = train_data[features]  # Your input features
    y = train_data['avgprice']  # Your target variable

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32 , shuffle=True)


    years_test = X_test['year'].values
    years_train = X_train['year'].values

    # Feature Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Initialising the SVM
    model = SVR()
    param_grid = {'kernel': ['rbf'],
        'C': [1, 10, 100, 1000],
        'gamma': [0.01, 0.03, 0.05, 0.07, 0.1, 1, 'auto'],
        'epsilon': [0.01, 0.03, 0.05, 0.07, 0.1, 1]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

    # Fitting the SVM to the Training set
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    
    # Predicting the Test set results
    predictions_test = best_model.predict(X_test_scaled)
    predictions_train = best_model.predict(X_train_scaled)

    forecast_data = get_forecasted_fuel_prices()
    print(forecast_data.head())
    x_forecast = forecast_data[features]
    years_forecast = forecast_data['year'].values
    x_forecast = sc.transform(x_forecast)
    forecasted_prices = best_model.predict(x_forecast)

    # Print the test loss and test MAE
    test_mae = mean_absolute_error(y_test, predictions_test)
    train_mae = mean_absolute_error(y_train, predictions_train)
    print(f"Test MAE: {test_mae}")
    print(f"Train MAE: {train_mae}")

    print("Test Data Predictions:")
    for actual, predicted in zip(y_test, predictions_test):
        print(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}")

    # Print predictions and actuals for train data
    print("Train Data Predictions:")
    for actual, predicted in zip(y_train, predictions_train):
        print(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}")

    plot_predictions(predictions_test, y_test, "Test Data")
    plot_predictions(predictions_train, y_train, "Train Data")

    save_predictions_to_db(y_test, predictions_test, years_test, y_train, predictions_train, years_train, forecasted_prices, years_forecast)

def plot_predictions(predictions, actuals, title):
# Plotting
    plt.figure(figsize=(12, 6))
# Sort the actual and predicted values for plotting
    indices = np.argsort(actuals)
    sorted_actual = actuals.iloc[indices].values
    sorted_predictions = predictions[indices]
# Plotting the Actual vs Predicted Values
    plt.plot(sorted_actual, label='Actual Values', color='blue', marker='o')
    plt.plot(sorted_predictions, label='Predicted Values', color='red', linestyle='--', marker='x')
    plt.title(f'Actual vs Predicted Values SVM Model({title})')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

train_svm_model()