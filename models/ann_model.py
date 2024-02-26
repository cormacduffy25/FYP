import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Loading the Data 
df = pd.read_csv('estimated_average_selling_prices_fuel_sources (1).csv')

print(df.head())