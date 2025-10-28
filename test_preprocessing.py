import os
from src.data_preprocessing import load_and_clean_data

data_path = os.path.join(os.getcwd(), "data", "retail_customers.csv")
df = load_and_clean_data(data_path)

print(df.head())
