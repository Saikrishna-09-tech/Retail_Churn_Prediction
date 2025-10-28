import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def train_churn_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Encode categorical columns
    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])
    
    # Drop non-numeric columns
    X = df.drop(['Customer_ID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Training Complete | Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, "models/churn_model.pkl")
    print("💾 Model saved successfully to models/churn_model.pkl")
    
    return model
