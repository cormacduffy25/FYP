import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error 
import numpy as np
import matplotlib.pyplot as plt
def train_svm_model():

     # Loading the Data
    train_data = pd.read_csv('estimated_average_selling_prices_fuel_sources (1).csv')

    print(train_data.head())

    features = ['Year', 'coalprice','oilprice', 'gasprice', 'nuclearprice', 'hydroprice', 'windsolarprice', 'cokebreezeprice']
    X = train_data[features]  # Your input features
    y = train_data['avgprice']  # Your target variable

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialising the SVM
    model = SVR(kernel='linear')

    # Fitting the SVM to the Training set
    model.fit(X_train, y_train)

    # Predicting the Test set results
    predictions = model.predict(X_test)

    # Print the test loss and test MAE
    mae = mean_absolute_error(y_test, predictions)
    print(f"Test MAE: {mae}")

    print("Prediction vs Actual")
    for actual, predicted in zip(np.array(y_test), predictions):
        print(f"Actual: {actual}, Predicted: {predicted}")
    
      # Plotting
    plt.figure(figsize=(12, 6))

    # Sort the actual and predicted values for plotting
    indices = np.argsort(y_test)
    sorted_actual = y_test.iloc[indices].values
    sorted_predictions = predictions[indices]

    plt.plot(sorted_actual, label='Actual Values', color='blue', marker='o')
    plt.plot(sorted_predictions, label='Predicted Values', color='red', linestyle='--', marker='x')

    plt.title('Actual vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

        
train_svm_model()