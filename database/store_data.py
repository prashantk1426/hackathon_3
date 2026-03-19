from sqlalchemy import create_engine
import pandas as pd

print("Storing data into SQL database...")

engine = create_engine("sqlite:///student.db")

df = pd.read_csv("data/student_data.csv")
df.to_sql("students", engine, if_exists="replace", index=False)

print("Data stored successfully in student.db")
