from sqlalchemy import create_engine
from env_var import db_url
import pandas as pd

engine = create_engine(db_url)

def load_data_from_db():
   sql_query = 'SELECT * FROM fuel_sources'
   return pd.read_sql_query(sql_query, engine)
