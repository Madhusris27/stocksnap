import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Helper function to preprocess and feature engineer the dataset
def process_data(df):
    # Standardize column names
    df.columns = df.columns.astype(str).str.strip().str.upper()

    # Identify essential columns
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col), None)
    
    if not close_col:
        st.error("The dataset must contain a column for closing price (e.g., 'Close', 'Close Price').")
        st.stop()
    
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            df = df.sort_values(by=date_col)
            # Set Date as index for plotting
            df.set_index(date_col, inplace=True)
        except Exception as e:
            st.warning(f"Could not parse dates properly from '{date_col}': {e}")
    else:
         st.warning("No Date column found. Proceeding using row index/order.")

    # Remove extra spaces or handle string-encoded numbers in close price
    if df[close_col].dtype == object:
        df[close_col] = df[close_col].str.replace(',', '').astype(float)
    
    # Handle missing values
    df = df.dropna(subset=[close_col])
    
    # Generate Target Column: 1 if Next Day Close > Current Day Close, else 0
    df['Target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
    # The last row's Target will technically be False (0) because shift(-1) is NaN.
    
    # Create simple Technical Indicators as Features
    df['SMA_10'] = df[close_col].rolling(window=10).mean()
    df['SMA_30'] = df[close_col].rolling(window=30).mean()
    df['Returns'] = df[close_col].pct_change()
    
    # Drop rows with NaN introduced by rolling/shifting except the last row
    # We only drop rows where our features are NaN
    df = df.dropna(subset=['SMA_10', 'SMA_30', 'Returns', close_col])
    
    return df, close_col

# Main App Execution
st.title("📈 Stock Market Predictor")
st.write("Upload your dataset (CSV) to train a Random Forest classifier and predict if the stock will go UP or DOWN the next day.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # 1. Load Data
    st.subheader("Data Preview")
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.dataframe(raw_df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
    df, close_col = process_data(raw_df)
    
    st.subheader("Processed Data (with Features and Target)")
    st.dataframe(df.head())

    # 2. Extract Features and Target
    features = ['SMA_10', 'SMA_30', 'Returns', close_col]
    X = df[features]
    y = df['Target']
    
    # Drop the very last row for training (since its target is based on a future day we don't have)
    X_train_data = X.iloc[:-1]
    y_train_data = y.iloc[:-1]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42, shuffle=False)

    # 3. Train the Model
    st.subheader("Model Training")
    with st.spinner("Training Random Forest Classifier..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # 4. Predict and Evaluate Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Random Forest Model Accuracy:** {accuracy * 100:.2f}%")

    # 5. Make Next-Day Prediction
    st.subheader("Prediction for the Next Trading Day")
    # Use the absolute last row of data for prediction
    last_row = X.iloc[[-1]] 
    prediction = model.predict(last_row)[0]
    
    # The actual Last Closing Price
    last_price = df[close_col].iloc[-1]
    
    st.write(f"**Latest Closing Price:** {last_price:.2f}")
    if prediction == 1:
        st.success("### 📈 Prediction: BUY (Price expected to increase)")
    else:
        st.error("### 📉 Prediction: NOT BUY (Price expected to decrease/stay flat)")

    # 6. Visualization
    st.subheader("Stock Price Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if isinstance(df.index, pd.DatetimeIndex):
        ax.plot(df.index, df[close_col], label='Closing Price', color='blue')
        ax.plot(df.index, df['SMA_10'], label='10-Day SMA', color='orange', linestyle='--')
        ax.plot(df.index, df['SMA_30'], label='30-Day SMA', color='red', linestyle='--')
        ax.set_xlabel("Date")
    else:
        ax.plot(df[close_col].values, label='Closing Price', color='blue')
        ax.plot(df['SMA_10'].values, label='10-Day SMA', color='orange', linestyle='--')
        ax.plot(df['SMA_30'].values, label='30-Day SMA', color='red', linestyle='--')
        ax.set_xlabel("Time Step")
        
    ax.set_ylabel("Price")
    ax.set_title("Nifty 50 Closing Price & Moving Averages")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Awaiting CSV file to be uploaded.")
