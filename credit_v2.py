import numpy as np
from river import tree, metrics
import random
import matplotlib.pyplot as plt

# Set constants
mean = 100
std_dev = 15  # Standard deviation for normal distribution
spike_prob = 0.01  # Probability of a large spike
low_threshold = 20  # Anomaly if below this
high_threshold = 200  # Anomaly if above this
num_transactions = 1000  # Number of transactions to generate

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
anomalies = []  # Track anomalous transactions
transactions = []  # Store transactions for visualization

for i in range(num_transactions):
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
        anomalies.append((i, value))
        print(f"Anomaly detected at index {i}: {value}")

# Plot the transactions and anomalies
plt.figure(figsize=(12, 6))
plt.plot(transactions, label='Transaction Value')

# Extract indices and values separately for anomalies
anomaly_indices = [index for index, _ in anomalies]
anomaly_values = [value for _, value in anomalies]

plt.scatter(anomaly_indices, anomaly_values, color='red', marker='x', label='Anomaly')
plt.axhline(low_threshold, color='green', linestyle='--', label='Low Threshold')
plt.axhline(high_threshold, color='purple', linestyle='--', label='High Threshold')
plt.legend()
plt.title('Transaction Stream with Anomalies')
plt.show()

print(f"Final Accuracy: {accuracy}")
