
# from flask import Flask, request, render_template, redirect, url_for, session, flash
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model 
# import os
# import plotly.graph_objs as go
# from plotly.offline import plot
# from datetime import timedelta, datetime
# import json

# app = Flask(__name__)
# app.secret_key = os.environ.get('SECRET_KEY', 'fallback_key_change_in_production')  # More secure secret key
# app.permanent_session_lifetime = timedelta(minutes=5)  # Extended session time

# # Dummy users for login
# users = {'arif': '9744503668', 'user1': 'mypassword'}

# # Load stock options
# stock_options = pd.read_csv("indian_stocks.csv")
# stock_list = stock_options.to_dict(orient="records")

# # ----------------- LOGIN ROUTE -----------------
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         if username in users and users[username] == password:
#             session.permanent = True
#             session['username'] = username
#             return redirect(url_for('home'))
#         else:
#             error = 'Invalid credentials. Try again.'
#     return render_template('login.html', error=error)

# # ----------------- LOGOUT ROUTE -----------------
# @app.route('/logout')
# def logout():
#     session.pop('username', None)
#     flash('✅ Logged out successfully.')
#     return redirect(url_for('login'))

# # ----------------- STOCK PREDICTION ROUTE -----------------
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if 'username' not in session:
#         return redirect(url_for('login'))

#     prediction = None
#     suggestion = None
#     avg_price = None
#     min_price = None
#     max_price = None
#     chart_html = None
#     selected_symbol = ""
#     selected_name = ""
#     company_info = {}
#     accuracy_info = {}
#     historical_prediction = None

#     if request.method == 'POST':
#         selected_symbol = request.form['stock']
#         selected_name = next((s['name'] for s in stock_list if s['symbol'] == selected_symbol), selected_symbol)

#         try:
#             model_path = f"models/{selected_symbol}_lstm_model.h5"
#             scaler_path = f"models/{selected_symbol}_scaler.pkl"
#             prediction_path = f"predictions/{selected_symbol}_prediction.json"

#             if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#                 prediction = f"❌ Model or scaler not found for {selected_name}."
#             else:
#                 model = load_model(model_path)
#                 scaler = joblib.load(scaler_path)

#                 input_data, avg_price, min_price, max_price = prepare_input_data(selected_symbol, scaler)

#                 if input_data is None:
#                     prediction = f"❌ Not enough data for {selected_name}."
#                 else:
#                     chart_html = generate_price_chart(selected_symbol)
                    
#                     # Load historical prediction if exists
#                     if os.path.exists(prediction_path):
#                         with open(prediction_path, 'r') as f:
#                             historical_prediction = json.load(f)

#                     # Make new prediction
#                     pred = model.predict(input_data)[0][0]
#                     prediction = f"Prediction for {selected_name}: {'📈 UP' if pred > 0.5 else '📉 DOWN'}"

#                     if pred > 0.7:
#                         suggestion = "💰 Strong BUY Signal"
#                     elif pred > 0.5:
#                         suggestion = "🟢 Consider Buying"
#                     elif pred < 0.3:
#                         suggestion = "🔻 Strong SELL Signal"
#                     else:
#                         suggestion = "🟠 Hold / Watch Closely"

#                     ticker_info = yf.Ticker(selected_symbol).info
#                     company_info = {
#                         'name': ticker_info.get('longName', 'N/A'),
#                         'industry': ticker_info.get('industry', 'N/A'),
#                         'market_cap': ticker_info.get('marketCap', 'N/A'),
#                         'pe_ratio': ticker_info.get('trailingPE', 'N/A'),
#                         'website': ticker_info.get('website', 'N/A')
#                     }

#                     # Load accuracy information
#                     acc_file = 'models/training_accuracy.csv'
#                     if os.path.exists(acc_file):
#                         acc_df = pd.read_csv(acc_file)
#                         acc_row = acc_df[acc_df['stock_symbol'] == selected_symbol]
#                         if not acc_row.empty:
#                             accuracy_info = {
#                                 'train_acc': acc_row.iloc[0]['train_accuracy'],
#                                 'val_acc': acc_row.iloc[0]['val_accuracy']
#                             }

#         except Exception as e:
#             prediction = f"❌ Error: {str(e)}"

#     return render_template('index.html',
#                            prediction=prediction,
#                            suggestion=suggestion,
#                            avg_price=avg_price,
#                            min_price=min_price,
#                            max_price=max_price,
#                            chart_html=chart_html,
#                            stock_options=stock_list,
#                            selected_symbol=selected_symbol,
#                            company_info=company_info,
#                            accuracy_info=accuracy_info,
#                            historical_prediction=historical_prediction)

# # ----------------- PREPARE DATA FUNCTION -----------------
# def prepare_input_data(stock_symbol, scaler):
#     df = yf.download(stock_symbol, period="20d", interval="15m")
#     if len(df) < 10:
#         return None, None, None, None

#     df['Return'] = df['Close'].pct_change()
#     df['MA5'] = df['Close'].rolling(window=5).mean()
#     df['MA10'] = df['Close'].rolling(window=10).mean()
#     df['Volatility'] = df['Close'].rolling(window=5).std()
#     df.dropna(inplace=True)

#     features = ['Close', 'Return', 'MA5', 'MA10', 'Volatility']
#     latest_data = df[features].iloc[-10:]
    
#     avg_close_price = round(float(latest_data['Close'].mean()), 2)
#     min_price = round(float(latest_data['Close'].min()), 2)
#     max_price = round(float(latest_data['Close'].max()), 2)

#     scaled_data = scaler.transform(latest_data)
#     input_array = np.array(scaled_data).reshape(1, 10, 5)

#     return input_array, avg_close_price, min_price, max_price

# # ----------------- GENERATE INTERACTIVE CHART -----------------
# def generate_price_chart(stock_symbol):
#     df = yf.download(stock_symbol, period="10d", interval="1h")
#     if df.empty:
#         return None

#     df['MA5'] = df['Close'].rolling(window=5).mean()
#     df['MA10'] = df['Close'].rolling(window=10).mean()
#     df.reset_index(inplace=True)

#     fig = go.Figure()

#     fig.add_trace(go.Candlestick(
#         x=df['Datetime'],
#         open=df['Open'],
#         high=df['High'],
#         low=df['Low'],
#         close=df['Close'],
#         name='Candlestick'
#     ))

#     fig.add_trace(go.Scatter(
#         x=df['Datetime'],
#         y=df['MA5'],
#         mode='lines',
#         line=dict(color='orange', width=2),
#         name='MA5'
#     ))

#     fig.add_trace(go.Scatter(
#         x=df['Datetime'],
#         y=df['MA10'],
#         mode='lines',
#         line=dict(color='blue', width=2),
#         name='MA10'
#     ))

#     fig.update_layout(
#         title=f'{stock_symbol} Candlestick Chart with MA5 & MA10',
#         xaxis_title='Date',
#         yaxis_title='Price',
#         xaxis_rangeslider_visible=False,
#         template='plotly_dark',
#         height=500
#     )

#     chart_html = plot(fig, output_type='div', include_plotlyjs=False)
#     return chart_html

# # ----------------- START APP -----------------
# if __name__ == '__main__':
#     if not os.path.exists("static"):
#         os.makedirs("static")
#     if not os.path.exists("models"):
#         os.makedirs("models")
#     if not os.path.exists("predictions"):
#         os.makedirs("predictions")
#     app.run(debug=True, port=5050)



from flask import Flask, request, render_template, redirect, url_for, session, flash
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model 
import os
import plotly.graph_objs as go
from plotly.offline import plot
from datetime import timedelta, datetime
import json
from plotly.subplots import make_subplots


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_key_change_in_production')  # More secure secret key
app.permanent_session_lifetime = timedelta(minutes=5)  # Extended session time

# Dummy users for login
users = {'arif': '9744503668', 'user1': 'mypassword'}

# Load stock options
stock_options = pd.read_csv("indian_stocks.csv")
stock_list = stock_options.to_dict(orient="records")

# ----------------- LOGIN ROUTE -----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session.permanent = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error = 'Invalid credentials. Try again.'
    return render_template('login.html', error=error)

# ----------------- LOGOUT ROUTE -----------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('✅ Logged out successfully.')
    return redirect(url_for('login'))

# ----------------- STOCK PREDICTION ROUTE -----------------
@app.route('/', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = None
    suggestion = None
    avg_price = None
    min_price = None
    max_price = None
    chart_html = None
    selected_symbol = ""
    selected_name = ""
    company_info = {}
    accuracy_info = {}
    historical_prediction = None
    market_summary = None

    if request.method == 'POST':
        selected_symbol = request.form['stock']
        selected_name = next((s['name'] for s in stock_list if s['symbol'] == selected_symbol), selected_symbol)

        try:
            model_path = f"models/{selected_symbol}_lstm_model.h5"
            scaler_path = f"models/{selected_symbol}_scaler.pkl"
            prediction_path = f"predictions/{selected_symbol}_prediction.json"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                prediction = f"❌ Model or scaler not found for {selected_name}."
            else:
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)

                input_data, avg_price, min_price, max_price = prepare_input_data(selected_symbol, scaler)

                if input_data is None:
                    prediction = f"❌ Not enough data for {selected_name}."
                else:
                    chart_html = generate_price_chart(selected_symbol)
                    
                    # Load historical prediction if exists
                    if os.path.exists(prediction_path):
                        with open(prediction_path, 'r') as f:
                            historical_prediction = json.load(f)

                    # Make new prediction
                    pred = model.predict(input_data)[0][0]
                    prediction = f"Prediction for {selected_name}: {'📈 UP' if pred > 0.5 else '📉 DOWN'}"

                    if pred > 0.7:
                        suggestion = "💰 Strong BUY Signal"
                    elif pred > 0.5:
                        suggestion = "🟢 Consider Buying"
                    elif pred < 0.3:
                        suggestion = "🔻 Strong SELL Signal"
                    else:
                        suggestion = "🟠 Hold / Watch Closely"

                    ticker_info = yf.Ticker(selected_symbol).info
                    company_info = {
                        'name': ticker_info.get('longName', 'N/A'),
                        'industry': ticker_info.get('industry', 'N/A'),
                        'market_cap': ticker_info.get('marketCap', 'N/A'),
                        'pe_ratio': ticker_info.get('trailingPE', 'N/A'),
                        'website': ticker_info.get('website', 'N/A')
                    }

                    # Load accuracy information
                    acc_file = 'models/training_accuracy.csv'
                    if os.path.exists(acc_file):
                        acc_df = pd.read_csv(acc_file)
                        acc_row = acc_df[acc_df['stock_symbol'] == selected_symbol]
                        if not acc_row.empty:
                            accuracy_info = {
                                'train_acc': acc_row.iloc[0]['train_accuracy'],
                                'val_acc': acc_row.iloc[0]['val_accuracy']
                            }

                    # Get market summary data
                    market_summary = get_market_summary(selected_symbol)

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"

    return render_template('index.html',
                           prediction=prediction,
                           suggestion=suggestion,
                           avg_price=avg_price,
                           min_price=min_price,
                           max_price=max_price,
                           chart_html=chart_html,
                           stock_options=stock_list,
                           selected_symbol=selected_symbol,
                           company_info=company_info,
                           accuracy_info=accuracy_info,
                           historical_prediction=historical_prediction,
                           market_summary=market_summary)

# ----------------- GET MARKET SUMMARY FUNCTION -----------------
def get_market_summary(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1d", interval="1m")
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[0]
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        # Get 52-week high/low
        yearly_hist = stock.history(period="1y")
        if not yearly_hist.empty:
            fifty_two_week_high = yearly_hist['High'].max()
            fifty_two_week_low = yearly_hist['Low'].min()
        else:
            fifty_two_week_high = "N/A"
            fifty_two_week_low = "N/A"
            
        # Get today's open, high, low
        today_open = hist['Open'].iloc[0] if len(hist) > 0 else "N/A"
        today_high = hist['High'].max() if len(hist) > 0 else "N/A"
        today_low = hist['Low'].min() if len(hist) > 0 else "N/A"
        
        return {
            'current_price': round(current_price, 2),
            'prev_close': round(prev_close, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'today_open': round(today_open, 2) if today_open != "N/A" else today_open,
            'today_high': round(today_high, 2) if today_high != "N/A" else today_high,
            'today_low': round(today_low, 2) if today_low != "N/A" else today_low,
            'fifty_two_week_high': round(fifty_two_week_high, 2) if fifty_two_week_high != "N/A" else fifty_two_week_high,
            'fifty_two_week_low': round(fifty_two_week_low, 2) if fifty_two_week_low != "N/A" else fifty_two_week_low,
            'timestamp': datetime.now().strftime("%d %b, %I:%M %p %Z")
        }
    except Exception as e:
        print(f"Error getting market summary: {e}")
        return None

# ----------------- PREPARE DATA FUNCTION -----------------
def prepare_input_data(stock_symbol, scaler):
    df = yf.download(stock_symbol, period="20d", interval="15m")
    if len(df) < 10:
        return None, None, None, None

    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df.dropna(inplace=True)

    features = ['Close', 'Return', 'MA5', 'MA10', 'Volatility']
    latest_data = df[features].iloc[-10:]
    
    avg_close_price = round(float(latest_data['Close'].mean()), 2)
    min_price = round(float(latest_data['Close'].min()), 2)
    max_price = round(float(latest_data['Close'].max()), 2)

    scaled_data = scaler.transform(latest_data)
    input_array = np.array(scaled_data).reshape(1, 10, 5)

    return input_array, avg_close_price, min_price, max_price

# ----------------- GENERATE INTERACTIVE CHART -----------------
from plotly.subplots import make_subplots  # Add this import at the top of your file
def generate_price_chart(stock_symbol):
    df = yf.download(stock_symbol, period="15d", interval="1h")
    if df.empty:
        return None

    # Calculate indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['VolumeMA5'] = df['Volume'].rolling(window=5).mean()
    df.reset_index(inplace=True)

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price chart (top)
    fig.add_trace(go.Candlestick(
        x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df['Datetime'], y=df['MA5'], name='MA5',
        line=dict(color='orange', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Datetime'], y=df['MA10'], name='MA10',
        line=dict(color='blue', width=1.5)
    ), row=1, col=1)

    # Volume chart (bottom)
    colors = ['green' if close > open else 'red' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(go.Bar(
        x=df['Datetime'], y=df['Volume'], name='Volume',
        marker_color=colors, opacity=0.7
    ), row=2, col=1)

    # Volume moving average
    fig.add_trace(go.Scatter(
        x=df['Datetime'], y=df['VolumeMA5'], name='Vol MA5',
        line=dict(color='black', width=1.5)
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'{stock_symbol} Price and Volume',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return plot(fig, output_type='div', include_plotlyjs=False)
# ----------------- START APP -----------------
if __name__ == '__main__':
    if not os.path.exists("static"):
        os.makedirs("static")
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    app.run(debug=True, port=5050)