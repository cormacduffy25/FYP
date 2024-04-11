from sqlalchemy import create_engine
from env_var import db_url
import pandas as pd

engine = create_engine(db_url)

def load_data_from_db():
   sql_query = 'SELECT * FROM estimated_average_selling_prices_fuel_sources'
   return pd.read_sql_query(sql_query, engine)
