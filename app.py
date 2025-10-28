import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Retail Churn Prediction", layout="wide")
st.title("🛍️ Retail Shop Customer Churn Prediction System")
st.write("Upload your customer data to predict who is likely to stop purchasing and get retention insights.")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # ---------- Data Loading ----------
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Customer Data")
    st.dataframe(df.head())

    # ---------- Preprocessing ----------
    encoder = LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])

    # Drop unnecessary column
    X = df.drop(['Customer_ID', 'Churn'], axis=1, errors='ignore')

    # ---------- Load Trained Model ----------
    model = joblib.load("models/churn_model.pkl")

    # ---------- Predict Churn ----------
    churn_prob = model.predict_proba(X)[:, 1]
    df['Churn_Probability'] = churn_prob

    # ---------- Display Predictions ----------
    st.subheader("🔍 Churn Prediction Results")
    st.dataframe(df[['Customer_ID', 'Churn_Probability']].sort_values(by='Churn_Probability', ascending=False))

    # ---------- Visualization Section ----------
    st.subheader("📈 Data Insights & Visualizations")

    # Churn probability distribution
    st.write("**Churn Probability Distribution**")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Churn_Probability'], bins=5, kde=True, color='orange', ax=ax1)
    st.pyplot(fig1)

    # Average spend vs churn probability
    if 'Total_Spend' in df.columns:
        st.write("**Average Spend vs Churn Probability**")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df['Total_Spend'], y=df['Churn_Probability'], hue=df['Churn_Probability'], palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

    # Visit frequency vs churn probability
    if 'Visit_Frequency' in df.columns:
        st.write("**Visit Frequency vs Churn Probability**")
        fig3, ax3 = plt.subplots()
        sns.barplot(x=df['Visit_Frequency'], y=df['Churn_Probability'], palette="viridis", ax=ax3)
        st.pyplot(fig3)

    # ---------- Download Predictions Section ----------
    st.subheader("💾 Download Predicted Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv,
        file_name='churn_predictions.csv',
        mime='text/csv'
    )

    # ---------- Retention Recommendations ----------
    st.subheader("💡 Business Retention Recommendations")
    st.write("""
    - Customers with churn probability **> 0.7** → Send loyalty offers or personalized discounts.  
    - Customers with churn probability **0.4–0.7** → Encourage them with reward points or thank-you messages.  
    - Customers **below 0.4** → Promote referral programs to increase engagement.  
    """)

else:
    st.info("Please upload a customer dataset to start predictions.")
    # ---------- Revenue Model Section ----------
    st.markdown("---")
    st.subheader("💼 Revenue Model")
    st.write("""
    **1️⃣ Subscription Model:**  
    Retailers pay a small monthly fee to access churn prediction and customer insights.

    **2️⃣ Commission Model:**  
    We partner with marketing and loyalty services, earning a 10% commission 
    for every successful customer re-engagement.
    """)
