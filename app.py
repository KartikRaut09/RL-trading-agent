import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces

# 🎮 Custom Gym Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.initial_balance = 10000.0
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_asset = self.balance
        return self._next_obs()

    def _next_obs(self):
        row = self.df.iloc[self.step_idx][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        max_val = np.max(np.abs(row)) if np.max(np.abs(row)) != 0 else 1.0
        norm_row = row / max_val
        return norm_row

    def step(self, action):
        price = float(self.df.iloc[self.step_idx]['Close'])
        reward = 0.0

        if action == 1 and self.position == 0.0:
            self.position = self.balance / price
            self.balance = 0.0
        elif action == 2 and self.position > 0.0:
            self.balance = self.position * price
            self.position = 0.0

        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        self.total_asset = self.balance + self.position * price
        reward = self.total_asset - self.initial_balance

        return self._next_obs(), reward, done, {}

# 🌐 Streamlit Dashboard
st.set_page_config(page_title="RL Trading Agent", layout="wide")
st.title("🤖 Reinforcement Learning Trading Agent Dashboard")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, BTC-USD)", value="AAPL")
start = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Train Agent"):
    st.write("⏳ Downloading data and training...")

    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    if df.empty:
        st.error("No data found. Try a different ticker or date range.")
    else:
        env = DummyVecEnv([lambda: TradingEnv(df)])
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=10000)

        # Simulate agent
        test_env = TradingEnv(df)
        obs = test_env.reset()
        portfolio_values = []
        actions = []
        prices = []
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = test_env.step(action)
            portfolio_values.append(test_env.total_asset)
            actions.append(action)
            prices.append(df.iloc[test_env.step_idx]['Close'])

        buy_points = [i for i, a in enumerate(actions) if a == 1]
        sell_points = [i for i, a in enumerate(actions) if a == 2]

        # 📈 Plot portfolio
        st.subheader("📈 Portfolio Value Over Time")
        fig1, ax1 = plt.subplots()
        ax1.plot(portfolio_values, label='Portfolio Value', color='blue')
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Value ($)")
        ax1.grid()
        st.pyplot(fig1)

        # 📊 Plot Price with Buy/Sell
        st.subheader("📊 Trading Actions on Price Chart")
        fig2, ax2 = plt.subplots()
        ax2.plot(prices, label="Price", color='gray')
        ax2.scatter(buy_points, [prices[i] for i in buy_points], color='green', marker='^', label='Buy', s=100)
        ax2.scatter(sell_points, [prices[i] for i in sell_points], color='red', marker='v', label='Sell', s=100)
        ax2.legend()
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Price")
        ax2.grid()
        st.pyplot(fig2)

        # Final result
        st.success(f"✅ Final Portfolio Value: ${portfolio_values[-1]:.2f}")
