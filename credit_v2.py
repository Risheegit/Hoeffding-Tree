import numpy as np
from river import tree, metrics
import random
import matplotlib.pyplot as plt
import time

# Set constants
mean = 100
std_dev = 15  # Standard deviation for normal distribution
spike_prob = 0.05  # Increased probability of a large spike
low_threshold = 20  # Anomaly if below this
high_threshold = 200  # Anomaly if above this

# Initialize Hoeffding Tree and a metric for evaluation
model = tree.HoeffdingTreeClassifier()
accuracy = metrics.Accuracy()

# Function to simulate a transaction
def generate_transaction():
    """Generates a normal transaction with occasional spikes."""
    if random.random() < spike_prob:
        return np.random.normal(mean * 2, std_dev * 5)  # Large spike
    return np.random.normal(mean, std_dev)

# Stream transactions and detect anomalies
anomaly_indices = []  # Track indices of anomalous transactions
anomaly_values = []  # Track values of anomalous transactions
transactions = []  # Store transactions for visualization

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(12, 6))

try:
    while True:
        value = generate_transaction()
        transactions.append(value)

        # Flag anomalies if out of bounds
        is_anomaly = value < low_threshold or value > high_threshold

        # Train the model incrementally with dummy labels (1 if anomaly, 0 otherwise)
        y_pred = model.predict_one({'value': value})  # Predict before updating
        accuracy.update(is_anomaly, y_pred)  # Track model performance
        model.learn_one({'value': value}, is_anomaly)  # Update the model

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
        ax.set_title('Transaction Stream with Anomalies')
        plt.draw()
        plt.pause(0.1)  # Pause briefly to update the plot

except KeyboardInterrupt:
    print("Stream stopped by user.")
    plt.ioff()  # Turn off interactive mode
    plt.show()

print(f"Final Accuracy: {accuracy}")