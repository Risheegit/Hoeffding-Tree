import numpy as np
from river import tree, metrics
import random
import matplotlib.pyplot as plt
import time

# Setting intial parameters
initial_mean = 1000
std_dev = 150
spike_prob = 0.05 
low_threshold = 200  
high_threshold = 2000  
seasonality_period = 50  
drift_rate = 0.1  

# Initialize Hoeffding Tree and a metric for evaluation
model = tree.HoeffdingTreeClassifier()
accuracy = metrics.Accuracy()

# Function to simulate a transaction with seasonal variations and concept drift
def generate_transaction(transaction_count):
    seasonal_mean = initial_mean + 10 * np.sin(2 * np.pi * transaction_count / seasonality_period)
    drifting_mean = seasonal_mean + drift_rate * transaction_count 
    if random.random() < spike_prob:
        return np.random.normal(drifting_mean * 2, std_dev * 5)
    return np.random.normal(drifting_mean, std_dev)

# Stream transactions and detect anomalies
anomaly_indices = []
anomaly_values = []
transactions = [] 
transaction_count = 0 

# Set up the plot
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

try:
    while True:
        value = generate_transaction(transaction_count)
        transactions.append(value)
        transaction_count += 1

        # Flag anomalies if out of bounds
        is_anomaly = value < low_threshold or value > high_threshold

        # Train the model incrementally with dummy labels (1 if anomaly, 0 otherwise)
        y_pred = model.predict_one({'value': value}) 
        accuracy.update(is_anomaly, y_pred) 
        model.learn_one({'value': value}, is_anomaly) 

        # Store anomalies
        if is_anomaly:
            anomaly_indices.append(len(transactions) - 1)
            anomaly_values.append(value)
            print(f"Anomaly detected at index {len(transactions)-1}: {value}")

        # Update the plot
        ax.clear()
        ax.plot(transactions, label='Transaction Value')
        if anomaly_indices and anomaly_values:
            ax.scatter(anomaly_indices, anomaly_values, color='red', marker='x', label='Anomaly')
        ax.legend()
        ax.set_title('Transaction Stream with Anomalies, Seasonality, and Drift')
        plt.draw()
        plt.pause(0.1) 
        print(f"Final Accuracy: {accuracy}")

except KeyboardInterrupt:
    print("Stream stopped by user.")
    plt.ioff() 
    plt.show()
