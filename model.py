# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import joblib
# import os

# # Load Indian stock symbols
# stock_options = pd.read_csv("indian_stocks.csv")
# stock_symbols = stock_options['symbol'].tolist()

# # Create output directory
# if not os.path.exists("models"):
#     os.makedirs("models")

# for stock_symbol in stock_symbols:
#     print(f"\nTraining model for {stock_symbol}...")

#     try:
#         df = yf.download(stock_symbol, period='200d', interval='1d')
#         if df.empty or len(df) < 20:
#             print(f"❌ Not enough data for {stock_symbol}, skipping.")
#             continue

#         df['Return'] = df['Close'].pct_change()
#         df['MA5'] = df['Close'].rolling(window=5).mean()
#         df['MA10'] = df['Close'].rolling(window=10).mean()
#         df['Volatility'] = df['Close'].rolling(window=5).std()
#         df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
#         df.dropna(inplace=True)

#         features = ['Close', 'Return', 'MA5', 'MA10', 'Volatility']
#         scaler = MinMaxScaler()
#         scaled_features = scaler.fit_transform(df[features])

#         # Save scaler
#         joblib.dump(scaler, f'models/{stock_symbol}_scaler.pkl')

#         X_scaled = pd.DataFrame(scaled_features, columns=features, index=df.index)
#         y = df['Target'].values

#         sequence_length = 10
#         X_seq, y_seq = [], []

#         for i in range(len(X_scaled) - sequence_length):
#             X_seq.append(X_scaled.iloc[i:i + sequence_length].values)
#             y_seq.append(y[i + sequence_length])

#         X_seq = np.array(X_seq)
#         y_seq = np.array(y_seq)

#         if len(X_seq) < 10:
#             print(f"❌ Not enough sequence data for {stock_symbol}, skipping.")
#             continue

#         # Train-test split
#         split_idx = int(len(X_seq) * 0.8)
#         X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
#         y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

#         # Build LSTM model
#         model = Sequential([
#             LSTM(50, return_sequences=False, input_shape=(sequence_length, X_train.shape[2])),
#             Dropout(0.2),
#             Dense(1, activation='sigmoid')
#         ])

#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#         # EarlyStopping callback
#         early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#         # Train model with 200 epochs and EarlyStopping
#         history = model.fit(X_train, y_train, epochs=20, batch_size=16,
#                             validation_data=(X_test, y_test),
#                             callbacks=[early_stop], verbose=0)

#         model.save(f'models/{stock_symbol}_lstm_model.h5')

#         # Get final training & validation accuracy
#         final_train_acc = history.history['accuracy'][-1]
#         final_val_acc = history.history['val_accuracy'][-1]

#         # Save accuracy to CSV
#         accuracy_file = 'models/training_accuracy.csv'
#         data = {
#             'stock_symbol': stock_symbol,
#             'train_accuracy': round(final_train_acc, 4),
#             'val_accuracy': round(final_val_acc, 4)
#         }

#         if os.path.exists(accuracy_file):
#             acc_df = pd.read_csv(accuracy_file)
#             acc_df = acc_df[acc_df.stock_symbol != stock_symbol]  # Remove old entry if exists
#             acc_df = pd.concat([acc_df, pd.DataFrame([data])], ignore_index=True)
#         else:
#             acc_df = pd.DataFrame([data])

#         acc_df.to_csv(accuracy_file, index=False)
#         print(f"✅ Model saved for {stock_symbol}. 📊 Train Acc={final_train_acc:.2f}, Val Acc={final_val_acc:.2f}")

#     except Exception as e:
#         print(f"❌ Error processing {stock_symbol}: {e}")

# print("\nAll models trained.")


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from datetime import datetime

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

def prepare_data(df):
    """Prepare features for LSTM model"""
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df

def train_stock_model(stock_symbol):
    """Train LSTM model for a given stock"""
    print(f"\nTraining model for {stock_symbol}...")

    try:
        # Fetch live data
        df = yf.download(stock_symbol, period='200d', interval='1d')
        
        if df.empty or len(df) < 50:
            print(f"❌ Not enough data for {stock_symbol}, skipping.")
            return False

        # Prepare features
        df = prepare_data(df)

        features = ['Close', 'Return', 'MA5', 'MA10', 'Volatility']
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[features])

        # Save scaler
        joblib.dump(scaler, f'models/{stock_symbol}_scaler.pkl')

        X_scaled = pd.DataFrame(scaled_features, columns=features, index=df.index)
        y = df['Target'].values

        # Create sequences
        sequence_length = 10
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled.iloc[i:i + sequence_length].values)
            y_seq.append(y[i + sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Train-test split
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train model
        history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stop], verbose=0)

        # Save model
        model.save(f'models/{stock_symbol}_lstm_model.h5')

        # Save accuracy
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        accuracy_file = 'models/training_accuracy.csv'
        data = {
            'stock_symbol': stock_symbol,
            'train_accuracy': round(final_train_acc, 4),
            'val_accuracy': round(final_val_acc, 4),
            'date': datetime.now().strftime('%Y-%m-%d')
        }

        # Update accuracy CSV
        if os.path.exists(accuracy_file):
            acc_df = pd.read_csv(accuracy_file)
            acc_df = acc_df[acc_df.stock_symbol != stock_symbol]
            acc_df = pd.concat([acc_df, pd.DataFrame([data])], ignore_index=True)
        else:
            acc_df = pd.DataFrame([data])

        acc_df.to_csv(accuracy_file, index=False)
        print(f"✅ Model saved for {stock_symbol}. 📊 Train Acc={final_train_acc:.2f}, Val Acc={final_val_acc:.2f}")
        return True

    except Exception as e:
        print(f"❌ Error processing {stock_symbol}: {e}")
        return False

def predict_stock_price(stock_symbol):
    """Make prediction for a given stock"""
    try:
        # Load model and scaler
        model = load_model(f'models/{stock_symbol}_lstm_model.h5')
        scaler = joblib.load(f'models/{stock_symbol}_scaler.pkl')

        # Fetch latest data
        df = yf.download(stock_symbol, period='50d', interval='1d')
        
        if df.empty:
            print(f"❌ Could not fetch data for {stock_symbol}")
            return None

        # Prepare features
        df = prepare_data(df)

        features = ['Close', 'Return', 'MA5', 'MA10', 'Volatility']
        scaled_features = scaler.transform(df[features])

        # Create sequence for prediction
        X_pred = scaled_features[-10:].reshape(1, 10, len(features))
        
        # Predict
        prediction = model.predict(X_pred)[0][0]
        
        # Prepare prediction result
        result = {
            'stock_symbol': stock_symbol,
            'last_close_price': df['Close'].iloc[-1],
            'prediction_probability': float(prediction),
            'prediction_text': 'Bullish' if prediction > 0.5 else 'Bearish',
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save prediction
        pred_file = f'predictions/{stock_symbol}_prediction.json'
        pd.Series(result).to_json(pred_file)
        
        print(f"Prediction for {stock_symbol}: {result['prediction_text']} (Prob: {prediction:.2f})")
        return result

    except Exception as e:
        print(f"❌ Prediction error for {stock_symbol}: {e}")
        return None

def main():
    # Load stock symbols from CSV
    try:
        stock_options = pd.read_csv("indian_stocks.csv")
        stock_symbols = stock_options['symbol'].tolist()
    except Exception as e:
        print(f"Error reading stock symbols: {e}")
        return

    # Train models
    trained_stocks = []
    for symbol in stock_symbols:
        if train_stock_model(symbol):
            trained_stocks.append(symbol)

    # Make predictions
    predictions = []
    for symbol in trained_stocks:
        prediction = predict_stock_price(symbol)
        if prediction:
            predictions.append(prediction)

    # Save overall predictions
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('predictions/stock_predictions.csv', index=False)
        print("\n📊 Predictions saved to predictions/stock_predictions.csv")

if __name__ == "__main__":
    main()