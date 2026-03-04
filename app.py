import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf
import plotly.graph_objects as go
from textblob import TextBlob
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Helper function to preprocess and feature engineer the dataset
def process_data(df):
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, infer_datetime_format=True)
        except Exception as e:
            st.warning(f"Could not parse dates from index: {e}")
    
    # Identify close column (yfinance uses Capitalized like 'Close')
    close_col = 'Close'
    if close_col not in df.columns:
        close_col = next((col for col in df.columns if 'CLOSE' in col.upper()), None)
        if not close_col:
            st.error("The dataset must contain a 'Close' column.")
            st.stop()

    # Handle missing values
    df = df.dropna(subset=[close_col])
    
    # Generate Target Column: 1 if Next Day Close > Current Day Close, else 0
    df['Target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
    
    # Technical Indicators as Features
    df['SMA_10'] = df[close_col].rolling(window=10).mean()
    df['SMA_30'] = df[close_col].rolling(window=30).mean()
    df['Returns'] = df[close_col].pct_change()
    
    # Add new indicators using 'ta' library
    rsi_indicator = RSIIndicator(close=df[close_col], window=14)
    df['RSI_14'] = rsi_indicator.rsi()
    
    macd_indicator = MACD(close=df[close_col])
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    
    bb_indicator = BollingerBands(close=df[close_col], window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    
    # Drop rows with NaN introduced by rolling/shifting
    features_to_check = ['SMA_10', 'SMA_30', 'Returns', 'RSI_14', 'MACD', 'BB_High', close_col]
    df = df.dropna(subset=features_to_check)
    
    return df, close_col

# Main App Execution
st.title("📈 Stock Market Predictor")
st.write("Enter a stock ticker to train a Random Forest classifier and predict if the stock will go UP or DOWN the next day.")

symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, ^NSEI, INFY.NS)", value="AAPL")

if symbol:
    # 1. Load Data
    st.subheader(f"Live Data Fetching for {symbol}")
    with st.spinner("Fetching data from Yahoo Finance..."):
        try:
            ticker = yf.Ticker(symbol)
            # Fetch 2 years of daily data
            raw_df = ticker.history(period="2y")
            
            if raw_df.empty:
                st.error(f"No data found for symbol '{symbol}'.")
                st.stop()
                
            st.dataframe(raw_df.tail())
            
            # Fetch Recent News for Sentiment
            news = ticker.news
            if news:
                st.subheader("Recent News Sentiment")
                sentiments = []
                for article in news[:5]:
                    title = article.get('title', '')
                    blob = TextBlob(title)
                    sentiments.append(blob.sentiment.polarity)
                    st.write(f"- {title} (Score: {blob.sentiment.polarity:.2f})")
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                st.write(f"**Average News Sentiment Score:** {avg_sentiment:.2f}")
            else:
                st.info("No recent news found for sentiment analysis.")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
        
    df, close_col = process_data(raw_df)
    
    st.subheader("Processed Data (with Features and Target)")
    st.dataframe(df.head())

    # 2. Extract Features and Target
    features = ['SMA_10', 'SMA_30', 'Returns', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', close_col]
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
    st.subheader("Interactive Stock Price Visualization")
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'] if 'Open' in df.columns else df[close_col],
                high=df['High'] if 'High' in df.columns else df[close_col],
                low=df['Low'] if 'Low' in df.columns else df[close_col],
                close=df[close_col],
                name='Price'))
                
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], line=dict(color='orange', width=1.5), name='10-Day SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], line=dict(color='red', width=1.5), name='30-Day SMA'))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=1, dash='dash'), name='BB High'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1, dash='dash'), name='BB Low', fill='tonexty'))

    fig.update_layout(title=f"{symbol} Stock Price & Indicators",
                      yaxis_title='Price',
                      xaxis_title='Date',
                      xaxis_rangeslider_visible=False,
                      template="plotly_dark")
                      
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please enter a stock ticker to begin.")
