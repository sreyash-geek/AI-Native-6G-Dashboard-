**AI-Native 6G Live Dashboard for Real-Time Monitoring**

In this project, I created a fully automated 6G network. A ML model trains on synthetically generated data initially. The data contains network metrics like signal strength, latency, bandwidth, network load, packet loss,
which are used to predict traffic congestion levels. The data is fed to LLMs for insights on the data and possible optimizations. The LLM insights and the state of the network is input to the RL controller agent, which
adaptively adjusts power to balance latency and optimize network parameters. The LLM insights, congestion predictions and the agent action is sent to the LLm for further diagnostics on possible errors and analysis.

The ML model is trained every 15 data points to make the model generalize well on new data and learn network dynamics better to form a closed-loop continuous learning phase.

To run the LIVE web-based simulation:
> streamlit run 6GAI.py
