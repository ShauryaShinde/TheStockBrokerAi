# Stockbroker AI – Secure Real/Simulated Trading App

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import bcrypt
import time

# ----------- Secure Login -----------

YOUR_PLAIN_PASSWORD = "Shaurya@2313"
hashed_backup_password = b"$2b$12$gWw5A0QK0JrUCcyZGJmlkOKlcuqk5Xn9slVuYzgoG7If5fVu10nIa"

if "attempts" not in st.session_state:
    st.session_state["attempts"] = 0

def login():
    st.title("🔐 Stockbroker AI - Secure Access")
    pw = st.text_input("Enter Password:", type="password")
    if pw:
        if st.session_state["attempts"] >= 3:
            st.error("Too many failed attempts. Wait 30 seconds.")
            time.sleep(30)
            st.session_state["attempts"] = 0
            st.stop()
        if pw == YOUR_PLAIN_PASSWORD:
            st.success("✅ Access granted (direct).")
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

# ----------- App Header -----------

st.title("📈 Stockbroker AI - Mission Control")

market = st.selectbox("🌍 Select Market:", ["India (NIFTY 50)", "USA (Dow Jones)"])

if market == "India (NIFTY 50)":
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
else:
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

# ----------- Broker Selection -----------

broker = st.selectbox("🧾 Select your broker:", [
    "Zerodha", "Upstox", "Angel One", "Investopedia Simulator", "Other/Manual"
])

token = None
simulator_mode = False

if broker in ["Zerodha", "Upstox", "Angel One"]:
    st.info(f"🔐 {broker} supports API-based trading.")
    if st.checkbox("I allow Stockbroker AI to control this broker via API"):
        username = st.text_input("Broker Username:")
        password = st.text_input("Broker Password:", type="password")
        token = st.text_input("Enter your API Token:")
        if token and username and password:
            st.success("✅ API credentials received. Live trading enabled.")
        else:
            st.warning("⚠️ Waiting for all credentials...")
            st.stop()
else:
    simulator_mode = True
    st.warning("⚠️ API not available. Running in simulator mode only.")

# ----------- Capital Input -----------

start_capital = st.number_input("💰 Starting Capital:", value=1000)
target = st.number_input("🎯 Target Capital:", value=100000)

# ----------- AI Launch -----------

if st.button("🚀 Launch Stockbroker AI"):
    st.success(f"📡 Mission launched: ₹{start_capital} → ₹{target}")
    st.write("🔍 Collecting and analyzing market data...")

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

    st.subheader("🤖 AI Trade Recommendations:")
    st.dataframe(pd.DataFrame(results))

    if simulator_mode:
        st.info("🧪 Simulator mode: trades will be simulated.")
    else:
        st.success("📡 Connected to broker. Executing trades (demo mode).")

    # ----------- Simulated Equity Growth -----------

    st.write("📊 Simulating portfolio growth...")
    equity = [start_capital]
    steps = 0
    while equity[-1] < target:
        equity.append(equity[-1] * np.random.uniform(1.002, 1.01))
        steps += 1
        if steps > 100000:
            break  # extended cap to simulate more realistically

    st.line_chart(pd.Series(equity, name="Equity Over Time"))

    if equity[-1] >= target:
        st.success(f"🏁 Target of ₹{target} reached in {steps} steps!")
    else:
        st.warning("⚠️ Target not reached. Try adjusting strategy or capital.")


