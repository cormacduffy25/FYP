import pandas as pd
from sklearn.model_selection import train_test_split
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

def save_predictions_to_db(actuals, predictions, years):
    df = pd.DataFrame({
        'actual': actuals, 
        'prediction': predictions, 
        'year': years})    
    df.to_sql('svm_prediction', con=engine, if_exists='replace', index=False)

def train_svm_model():

     # Loading the Data
    train_data = load_data_from_db()

    print(train_data.head())

    features = ['year', 'coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']
    X = train_data[features]  # Your input features
    y = train_data['avgprice']  # Your target variable

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)


    years_test = X_test['year'].values

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialising the SVM
    model = SVR(kernel='linear')

    # Fitting the SVM to the Training set
    model.fit(X_train, y_train)

    # Predicting the Test set results
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)

    # Print the test loss and test MAE
    test_mae = mean_absolute_error(y_test, predictions_test)
    train_mae = mean_absolute_error(y_train, predictions_train)
    print(f"Test MAE: {test_mae}")
    print(f"Train MAE: {train_mae}")

    print("Prediction vs Actual")
    for actual, predicted in zip(np.array(y_test), predictions_test):
        print(f"Actual: {actual}, Predicted: {predicted}")
  
    save_predictions_to_db(y_test, predictions_test, years_test)

# Plotting
    plt.figure(figsize=(12, 6))
# Sort the actual and predicted values for plotting
    indices = np.argsort(y_test)
    sorted_actual = y_test.iloc[indices].values
    sorted_predictions = predictions_test[indices]
# Plotting the Actual vs Predicted Values
    plt.plot(sorted_actual, label='Actual Values', color='blue', marker='o')
    plt.plot(sorted_predictions, label='Predicted Values', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Values SVM Model')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
train_svm_model()