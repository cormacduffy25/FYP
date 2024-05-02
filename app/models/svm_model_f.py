import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine
import json

def get_db_url():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config['db_url']

db_url = get_db_url()
engine = create_engine(db_url)

def load_data_from_db():
    sql_query = 'SELECT * FROM fuel_sources_lagged'
    data = pd.read_sql_query(sql_query, engine)
    return data

# Load the dataset
data = load_data_from_db()

# Dropping rows with missing values
data_cleaned = data.dropna()

fuel_types = ['oilprice', 'coalprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice']
all_forecasts = {}

for fuel in fuel_types:
    # Generating lagged features for forecasting
    for lag in [1, 2, 3]:
        data_cleaned[f'{fuel}_lag{lag}'] = data_cleaned[fuel].shift(lag)
    
    data_model = data_cleaned.dropna()  # Drop rows with any NaN values

    # Features and target for the model
    features = [f'{fuel}_lag{lag}' for lag in [1, 2, 3]]
    X = data_model[features]
    y = data_model[fuel]

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the model and grid search parameters
    model = SVR()
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [1, 10, 100, 1000],
        'gamma': [0.01, 0.03, 0.05, 0.07, 0.1, 1, 'auto'],
        'epsilon': [0.01, 0.03, 0.05, 0.07, 0.1, 1]
    }

    # Setup the GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_scaled, y)

    # Best parameters and best score
    print(f"Best parameters for {fuel}:", grid_search.best_params_)
    print(f"Best cross-validation score for {fuel} (negative MSE):", grid_search.best_score_)

    # Use the best estimator to forecast future years
    best_model = grid_search.best_estimator_
    latest_scaled = scaler.transform(data.iloc[-1][features].values.reshape(1, -1))

    predictions = {}
    years_to_predict = [2023, 2024, 2025]
    current_features = latest_scaled

    for year in years_to_predict:
        forecast = best_model.predict(current_features)[0]
        predictions[year] = forecast
        current_features = scaler.transform([[forecast] + list(current_features[0, :-1])])

    all_forecasts[fuel] = predictions

forecast_df = pd.DataFrame(all_forecasts)
forecast_df.index.name = 'year'
forecast_df.reset_index(inplace=True)

forecast_df.to_sql('fuelsources', con=engine, if_exists='append', index=False)

print("Data successfully saved to the database.")

print(all_forecasts)
