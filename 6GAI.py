# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import time
import gym
from gym import spaces
import threading
import streamlit as st
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings("ignore", message="You provided an OpenAI Gym environment")
import os

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key is None:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

# Flag to control simulation termination
stop_simulation = False

# Data Storage 
simulation_data = {
    'current_sample': None,
    'ml_prediction': None,
    'rl_action': "--",
    'llm_insights': "--",
    'self_healing': "--"
}

# To store a history of optimization events for continuous learning updates
optimization_history = []  # Store Optimization Insights from each iteration

# Synthetic Network Traffic Data Simulation
def generate_network_data(num_points=1):
    signal_strength = np.random.uniform(-100, -30, num_points)  # dBm
    latency = 150 - (signal_strength + 100) + np.random.normal(0, 5, num_points)
    congestion_level = np.digitize(latency, bins=[50, 100])  # 0: Low, 1: Medium, 2: High
    packet_loss = np.clip(congestion_level * np.random.uniform(0.5, 5.0, num_points) +
                          np.random.normal(0, 1, num_points), 0, 10)
    bandwidth = np.clip(100 - (latency * 0.5) + np.random.normal(0, 3, num_points), 1, 100)
    df = pd.DataFrame({
        'signal_strength': signal_strength,
        'latency': latency,
        'packet_loss': packet_loss,
        'bandwidth': bandwidth,
        'congestion_level': congestion_level
    })
    return df

# Train a RandomForestRegressor on synthetic data
def train_ml_model_from_data(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print("ML Model trained on new data. Mean Absolute Error:", mae)
    return model

# Initial ML Model training using synthetic data (as a baseline)
def train_ml_model():
    data = generate_network_data(100)
    X = data[['signal_strength', 'latency', 'packet_loss', 'bandwidth']]
    y = data['congestion_level']
    model = train_ml_model_from_data(X, y)
    return model

ml_model = train_ml_model()

# Setup Gemini API for LLM Usage
genai.configure(api_key=gemini_api_key)
model_gen = genai.GenerativeModel("gemini-2.0-flash")

def gemini_insights(text_logs):
    response = model_gen.generate_content(
        f"Analyze the following network logs and suggest top most optimization (only if needed), else respond with '--' : {text_logs}"
    )
    return response.text

def gemini_learn(text_logs):
    response = model_gen.generate_content(
        f"Learn from these past experiences and optimizations for future network insights: {text_logs}"
    )
    return response.text

def self_healing_diagnosis(error_logs):
    response = model_gen.generate_content(
        f"Diagnose the following network failure logs and assuming that multi-agent AI agents are monitoring the network suggest top 3 actions to be taken by them only in case of alarming insights got from LLM + RL action taken else respond with -- (max 3 lines): {error_logs}"
    )
    return response.text

# Basic encoding for conversion to input state for RL Control Agent
def encode_llm_insight(insight):
    return 1.0 if "optimization" in insight.lower() else 0.0

# RL Simulation Env for Network Control
class NetworkEnv(gym.Env):
    def __init__(self):
        super(NetworkEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Decrease power, 1: Increase power, 2: Maintain
        self.state = np.random.rand(4)
        
    def step(self, action):
        if action == 0:
            self.state[0] = max(0, self.state[0] - 0.05)
        elif action == 1:
            self.state[0] = min(1, self.state[0] + 0.05)
        self.state[1] = np.clip(self.state[1] + np.random.normal(0, 0.02), 0, 1)
        reward = -abs(self.state[1] - 0.5)
        done = False
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.random.rand(4)
        return self.state

env = NetworkEnv()
# Define the RL agent using stable-baselines3 PPO
rl_agent = PPO("MlpPolicy", env, verbose=0, device = 'cpu')
rl_agent.learn(total_timesteps=2048)  

# Continuous Learning & Knowledge Retention Module
def continuous_learning():
    global optimization_history, ml_model
    print("\n=== Continuous Learning Phase ===")
    
    # Summarize past optimizations
    history_summary_text = "Summary of optimizations:\n"
    new_data = []
    new_labels = []
    for record in optimization_history:
        sample = record['sample']
        history_summary_text += (f"Iteration {record['iteration']}: "
                                 f"Latency {sample['latency']:.2f} ms, "
                                 f"ML Congestion {record['ml_prediction']}, "
                                 f"RL Action {record['rl_action']}, "
                                 f"Self-Healing: {record['self_healing']}\n")
        # Extract new training data from each sample
        new_data.append([sample['signal_strength'], sample['latency'], sample['packet_loss'], sample['bandwidth']])
        new_labels.append(sample['congestion_level'])
    
    summary = gemini_learn(history_summary_text)
    print("LLM Summary of Past Optimizations:")
    print(summary)
    
    # Retrain the ML model on the new data collected in the last 15 iterations
    if new_data:
        X_new = np.array(new_data)
        y_new = np.array(new_labels)
        ml_model = train_ml_model_from_data(X_new, y_new)
    print("Continuous learning loop completed: ML model updated.")

# Real-Time Simulation Loop
def run_real_time_system(iterations=100, interval=15):
    global simulation_data, ml_model, rl_agent, env, stop_simulation, optimization_history
    for i in range(iterations):
        if stop_simulation:
            print("Simulation stopped by user.")
            break
        
        data_df = generate_network_data(num_points=1)
        sample = data_df.iloc[0]
        simulation_data['current_sample'] = sample

        features_df = pd.DataFrame([[sample['signal_strength'], sample['latency'], sample['packet_loss'], sample['bandwidth']]],
                           columns=['signal_strength', 'latency', 'packet_loss', 'bandwidth'])
        ml_pred = ml_model.predict(features_df)[0]

        simulation_data['ml_prediction'] = ml_pred

        log_text = (f"Signal: {sample['signal_strength']:.2f}, Latency: {sample['latency']:.2f}, "
                    f"Packet Loss: {sample['packet_loss']:.2f}, Bandwidth: {sample['bandwidth']:.2f}, "
                    f"ML Congestion: {ml_pred}")
        llm_output = gemini_insights(log_text)
        simulation_data['llm_insights'] = llm_output

        normalized_signal = (sample['signal_strength'] + 100) / 70.0
        normalized_latency = sample['latency'] / 150.0
        normalized_ml = ml_pred / 2.0
        encoded_insight = encode_llm_insight(llm_output)
        combined_state = np.array([normalized_signal, normalized_latency, normalized_ml, encoded_insight], dtype=np.float32)
        
        # RL Control Agent - Adaptive Power Usage
        global rl_agent  
        action, _ = rl_agent.predict(combined_state)
        new_state, reward, _, _ = env.step(action)
        
        if action == 0:
            simulation_data['rl_action'] = "Decreasing"
        elif action == 1:
            simulation_data['rl_action'] = "Increasing"
        elif action == 2:
            simulation_data['rl_action'] = "Maintaining"
        
        simulation_data['rl_state'] = new_state

        if sample['latency'] > 120 or sample['packet_loss'] > 5 or (ml_pred >= 2 and action != 1):
            error_text = (f"Network state: Signal {sample['signal_strength']:.2f} dBm, "
                          f"Latency {sample['latency']:.2f} ms, Packet Loss {sample['packet_loss']:.2f}%, "
                          f"Bandwidth {sample['bandwidth']:.2f} Mbps; ML Congestion: {ml_pred}; "
                          f"LLM Insight: {llm_output}; RL Decision: {simulation_data['rl_action']}")
            simulation_data['self_healing'] = self_healing_diagnosis(error_text)
        else:
            simulation_data['self_healing'] = "--"
        
        optimization_history.append({
            'iteration': i,
            'sample': sample.to_dict(),
            'ml_prediction': ml_pred,
            'llm_insights': llm_output,
            'rl_action': simulation_data['rl_action'],
            'self_healing': simulation_data['self_healing'],
            'reward': reward
        })

        # Retrain ML model every 15 iterations using the new data collected
        if i % 15 == 0 and i > 0:
            continuous_learning()
            optimization_history.clear()
        
        time.sleep(interval)

# Streamlit Dashboard for Real-Time Visualization
def run_dashboard():
    st.set_page_config(page_title="6G Dashboard", layout="wide")
    st.title("Real-Time 6G Network Simulation Dashboard (Powered by AI)")
    st.write("---Network Metrics [LIVE]---")
    
    # Initialize or retrieve live history for latency graph
    if 'latency_history' not in st.session_state:
        st.session_state.latency_history = []
    if 'time_history' not in st.session_state:
        st.session_state.time_history = []
    
    # Placeholders for the metrics, graph, and AI suggestions & Diagnostics
    metrics_placeholder = st.empty()
    graph_placeholder = st.empty()
    ai_suggestions_placeholder = st.empty()
    
    start_time = time.time()
    while not stop_simulation:
        current_time = time.time() - start_time

        # Update metrics in the same place
        with metrics_placeholder.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            if simulation_data['current_sample'] is not None:
                sample = simulation_data['current_sample']
                
                # Calculate change in Signal Strength
                if 'prev_signal' not in st.session_state:
                    st.session_state['prev_signal'] = sample['signal_strength']
                delta_signal = sample['signal_strength'] - st.session_state['prev_signal']
                st.session_state['prev_signal'] = sample['signal_strength']
                
                # Calculate change in Latency
                if 'prev_latency' not in st.session_state:
                    st.session_state['prev_latency'] = sample['latency']
                delta_latency = sample['latency'] - st.session_state['prev_latency']
                st.session_state['prev_latency'] = sample['latency']
                
                # Calculate change in Packet Loss
                if 'prev_packet_loss' not in st.session_state:
                    st.session_state['prev_packet_loss'] = sample['packet_loss']
                delta_packet_loss = sample['packet_loss'] - st.session_state['prev_packet_loss']
                st.session_state['prev_packet_loss'] = sample['packet_loss']
                
                # Calculate change in Bandwidth
                if 'prev_bandwidth' not in st.session_state:
                    st.session_state['prev_bandwidth'] = sample['bandwidth']
                delta_bandwidth = sample['bandwidth'] - st.session_state['prev_bandwidth']
                st.session_state['prev_bandwidth'] = sample['bandwidth']
                
                col1.metric("Signal Strength (dBm)", f"{sample['signal_strength']:.2f}",
                            delta=f"{delta_signal:.2f} dBm")
                col2.metric("Latency (ms)", f"{sample['latency']:.2f}",
                            delta=f"{delta_latency:.2f} ms")
                col3.metric("Packet Loss (%)", f"{sample['packet_loss']:.2f}",
                            delta=f"{delta_packet_loss:.2f} %")
                col4.metric("Bandwidth (Mbps)", f"{sample['bandwidth']:.2f}",
                            delta=f"{delta_bandwidth:.2f} Mbps")
                col5.metric("Automated Action (Power)", f"{simulation_data['rl_action']}")
            else:
                st.write("...Waiting for network data...")
        
        # Update live latency graph with current time (Timestamp)
        with graph_placeholder.container():
            if simulation_data['current_sample'] is not None:
                sample = simulation_data['current_sample']
                st.session_state.latency_history.append(sample['latency'])
                st.session_state.time_history.append(current_time)
                if len(st.session_state.latency_history) > 50:
                    st.session_state.latency_history = st.session_state.latency_history[-50:]
                    st.session_state.time_history = st.session_state.time_history[-50:]
            latency_df = pd.DataFrame({
                "Time (s)": st.session_state.time_history,
                "Latency (ms)": st.session_state.latency_history
            })
            st.line_chart(latency_df.set_index("Time (s)"))
        
        # Update AI suggestions - LLM Insights, Self-Healing 
        with ai_suggestions_placeholder.container():
            st.subheader("AI Suggestions & Diagnostics:")
            st.markdown("**LLM Insights:**")
            st.write(simulation_data['llm_insights'])
            st.markdown("**Self-Healing Mechanism:**")
            st.write(simulation_data['self_healing'])
        
        time.sleep(20)


# Launch simulation and live dashboard mimicking a digital-twin
if __name__ == "__main__":
    simulation_thread = threading.Thread(target=run_real_time_system, args=(100, 20))
    simulation_thread.start()
    run_dashboard()
    simulation_thread.join()
