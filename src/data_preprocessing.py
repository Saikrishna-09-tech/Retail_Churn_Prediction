import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path):
    # Load dataset
    df = pd.read_csv(path)

    # Display basic info
    print("Dataset Loaded Successfully ✅")
    print("Shape of data:", df.shape)

    # Encode categorical column 'Gender'
    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])

    # Check for missing values
    if df.isnull().sum().any():
        print("⚠️ Missing values found, filling with mean values.")
        df.fillna(df.mean(), inplace=True)
    else:
        print("No missing values found ✅")

    return df
