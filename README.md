# 🤖 RL Trading Agent Dashboard

This project is a Streamlit web app for training a **Reinforcement Learning agent** (PPO) to trade stocks or cryptocurrencies using historical price data.

## 🚀 Features
- Train RL agents on any stock/crypto symbol (via Yahoo Finance)
- Visualize Buy/Sell actions and portfolio value
- Interactive Streamlit dashboard

## 📦 Tech Stack
- Streamlit for dashboard UI
- Stable-Baselines3 for reinforcement learning
- Gym for the custom trading environment
- YFinance for historical market data
- Matplotlib, Pandas, NumPy

## 🖥️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/KartikRaut09/rl-trading-agent.git
cd rl-trading-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 🔮 Future Enhancements
- Crypto data from Binance API
- Multi-stock portfolio training
- Strategy comparison dashboard
