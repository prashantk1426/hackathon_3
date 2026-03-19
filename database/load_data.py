from sqlalchemy import create_engine
import pandas as pd

def load_data():
    engine = create_engine("sqlite:///student.db")
    df = pd.read_sql("SELECT * FROM students", engine)
    return df
