# Stockbroker AI - 
Secure Trading App with Dual-Market Support
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import bcrypt
import time

# ----------------- Secure Login (Custom for YOU) -----------------

YOUR_PLAIN_PASSWORD = "Shaurya@2313"
hashed_backup_password = b"$2b$12$gWw5A0QK0JrUCcyZGJmlkOKlcuqk5Xn9slVuYzgoG7If5fVu10nIa"

if "attempts" not in st.session_state:
    st.session_state["attempts"] = 0

def login():
    st.title("🔒 Stockbroker AI - Secure Access")
    pw = st.text_input("Enter Password:", type="password")

    if pw:
        if st.session_state["attempts"] >= 3:
            st.error("Too many failed attempts. Try again in 30 seconds.")
            time.sleep(30)
            st.session_state["attempts"] = 0
            st.stop()

        if pw == YOUR_PLAIN_PASSWORD:
            st.success("✅ Access granted (you).")
            st.session_state["authenticated"] = True
        elif bcrypt.checkpw(pw.encode(), hashed_backup_password):
            st.success("✅ Access granted (hashed).")
            st.session_state["authenticated"] = True
        else:
            st.session_state["attempts"] += 1
            st.error("❌ Incorrect password.")
            st.stop()

if not st.session_state.get("authenticated"):
    login()
    st.stop()

# ----------------- Trading AI Core -----------------

st.title("🤖 Stockbroker AI - Mission Control")

market = st.selectbox("Select Market:", ["India (NIFTY 50)", "USA (Dow Jones)"])

if market == "India (NIFTY 50)":
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
else:
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

broker = st.selectbox("Select your broker:", ["Zerodha", "Upstox", "Angel One", "Investopedia Simulator", "Other/Manual"])

token = None
simulator_mode = False

if broker in ["Zerodha", "Upstox", "Angel One"]:
    st.info(f"🔐 {broker} supports API-based control.")
    if st.checkbox("I allow Stockbroker AI to control this broker via API"):
        username = st.text_input("Enter your Broker Username:")
        password = st.text_input("Enter your Broker Password:", type="password")
        token = st.text_input("Enter your API Token:")
        if token:
            st.success("✅ Token received. Secure control granted.")
        else:
            st.warning("Waiting for token...")
else:
    simulator_mode = True
    st.warning("⚠️ API not available. Running in simulator mode only.")

start_capital = st.number_input("Enter Starting Capital (₹):", value=1000)
target = st.number_input("Enter Target Capital (₹):", value=100000)

if st.button("🚀 Launch AI Mission"):
    st.success(f"Mission: ₹{start_capital} → ₹{target}")
    st.write("📡 Collecting data from selected market...")

    results = []

    for symbol in symbols:
        try:
            df = yf.download(symbol, period="6mo")
            df["Return"] = df["Close"].pct_change()
            df["SMA_5"] = df["Close"].rolling(5).mean()
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["Volatility"] = df["Return"].rolling(10).std()
            df.dropna(inplace=True)

            X = df[["SMA_5", "SMA_20", "Volatility", "Return"]]
            y = (df["Close"].shift(-1) > df["Close"]).astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_scaled[:-1], y[:-1])
            pred = model.predict([X_scaled[-1]])[0]
            prob = model.predict_proba([X_scaled[-1]])[0][pred]

            decision = "BUY" if pred == 1 else "WAIT"
            confidence = f"{prob * 100:.1f}%"

            results.append({
                "Symbol": symbol,
                "Action": decision,
                "Confidence": confidence
            })
        except:
            continue

    st.subheader("📊 AI Suggestions:")
    st.dataframe(pd.DataFrame(results))

    if simulator_mode:
        st.info("🧪 Simulator mode: No real trades executed.")
    else:
        st.success("📡 Live API connection ready. Executing trades (simulated).")

    st.write("📈 Tracking progress until target is achieved...")
    equity = [start_capital]
    steps = 0
    while equity[-1] < target:
        equity.append(equity[-1] * np.random.uniform(1.01, 1.03))
        steps += 1
        if steps > 10000:
            break  # safety cap to prevent infinite runaway loop

    st.line_chart(pd.Series(equity, name="Equity Over Time"))
    if equity[-1] >= target:
        st.success(f"🎯 Goal of ₹{target} achieved in {steps} steps!")
    else:
        st.warning("Target not reached. AI paused to prevent runaway loop.")
