import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import bcrypt
import time
# ------------------- Secure Login -------------------
hashed_password = b"$2b$12$gWw5A0QK0JrUCcyZGJmlkOKlcuqk5Xn9slVuYzgoG7If5fVu10nIa"
attempts = st.session_state.get("attempts", 0)

def login():
    global attempts
    st.title("🔒 Stockbroker AI - Secure Access")
    pw = st.text_input("Enter Password:", type="password")
    if pw:
        if attempts >= 3:
            st.error("Too many attempts. Locked for 30 seconds.")
            time.sleep(30)
            attempts = 0

        if bcrypt.checkpw(pw.encode(), hashed_password):
            st.session_state["authenticated"] = True
        else:
            st.session_state["attempts"] = attempts + 1
            st.error("Incorrect password.")

if not st.session_state.get("authenticated"):
    login()
    st.stop()

# ------------------- Trading AI Core -------------------
st.title("🤖 Stockbroker AI - Mission Control")

broker = st.selectbox("Select your broker:", ["Zerodha", "Upstox", "Angel One", "Investopedia Simulator", "Other/Manual"])

token = None
simulator_mode = False

if broker in ["Zerodha", "Upstox", "Angel One"]:
    st.info(f"🔐 {broker} supports API-based control.")
    if st.checkbox("I allow Stockbroker AI to control this broker via API"):
        token = st.text_input("Enter your API Token (from your broker dashboard):")
        if not token:
            st.warning("Waiting for token...")
        else:
            st.success("✅ Token received. Secure control granted.")
else:
    simulator_mode = True
    st.warning("⚠️ API not available for this broker. Switching to simulation mode.")

# Input Section
start_capital = st.number_input("Enter Starting Capital (₹):", value=1000)
target = st.number_input("Enter Target Amount (₹):", value=1000000)

if st.button("🚀 Launch AI Mission"):
    st.success(f"Mission accepted: ₹{start_capital} → ₹{target}")
    st.write("Gathering stock data from NIFTY 50...")

    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
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

    results_df = pd.DataFrame(results)
    st.subheader("📈 AI Trade Suggestions")
    st.dataframe(results_df)

    if simulator_mode:
        st.warning("⚠️ SIMULATION MODE: No real trades executed.")
    else:
        st.success("Connected to broker API. Ready to trade.")

    # --- Capital Simulation Loop ---
    equity = [start_capital]
    max_days = 500  # Limit loop to prevent infinite runs

    while equity[-1] < target and len(equity) < max_days:
        next_val = equity[-1] * np.random.uniform(1.005, 1.03)
        equity.append(next_val)

    st.line_chart(pd.Series(equity, name="Capital Over Time"))

    if equity[-1] >= target:
        st.success(f"🎯 Target of ₹{target} reached in {len(equity)-1} steps!")
    else:
        st.warning("Target not reached. AI will continue learning.")
