import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine
import json

# Function to load database configuration and connect to the engine
def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

db_url = get_db_url()
engine = create_engine(db_url)

def load_data_from_db():
    sql_query = 'SELECT * FROM fuel_sources_lagged'
    return pd.read_sql_query(sql_query, engine)

def save_forecasts_to_database(forecasts, engine, table_name, years):
    # Convert the forecasts dictionary to a DataFrame
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.index = years
    forecast_df.index.name = 'year'
    forecast_df.reset_index(inplace=True)
    
    # Save the DataFrame to SQL
    forecast_df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print("Data successfully saved to the database.")

def train_svm_model():
# Load and clean the dataset
    data = load_data_from_db()
    data.dropna(inplace=True)

    fuel_types = ['oilprice', 'coalprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']
    all_forecasts = {}

    for fuel in fuel_types:
        # Prepare lagged features for forecasting
        for lag in [1, 2, 3]:
            data[f'{fuel}_lag{lag}'] = data[fuel].shift(lag)
    
        data_model = data.dropna()

    # Define features and target
        features = [f'{fuel}_lag{lag}' for lag in [1, 2, 3]]
        x = data_model[features]
        y = data_model[fuel]

        # Scale the features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Define the model and grid search parameters
        model = SVR()
        param_grid = {
            'kernel': ['rbf'],
            'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.03, 0.05, 0.07, 0.1, 1, 'auto'],
            'epsilon': [0.01, 0.03, 0.05, 0.07, 0.1, 1]
        }

    # Setup the GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
        grid_search.fit(x_train, y_train)

    # Use the best estimator to make predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)

        print(f"Best parameters for {fuel}: {grid_search.best_params_}")
        print(f"Best cross-validation score for {fuel} (negative MSE): {grid_search.best_score_}")
        print(f"Test MAE for {fuel}: {test_mae}")
        print(f"Test MSE for {fuel}: {test_mse}")

    # Forecasting future values using recursive predictions
        latest_features = data.iloc[-1][features].values.reshape(1, -1)
        latest_scaled = scaler.transform(latest_features)
        predictions = {}
        years_to_predict = [2023, 2024, 2025]
        current_features = latest_scaled

        for year in years_to_predict:
            forecast = best_model.predict(current_features)[0]
            predictions[year] = forecast
            current_features = scaler.transform(np.array([[forecast] + list(current_features[0, :-1])]))

        all_forecasts[fuel] = predictions
    save_forecasts_to_database(all_forecasts, engine, 'fuelsources_forecasted_svm', years_to_predict)
    print(all_forecasts)
