import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Generate Time-Series Data
def generate_time_series():
    x = np.arange(1, 51)  # Time steps from 1 to 50
    k = 1.5  # Power-law exponent
    y = 50 + 300 * (x / 50) ** k  # Power-law function
    noise = np.random.normal(0, 3, len(x))  # Add Gaussian noise
    return x, y + noise


# Calculate Prediction Interval (95% Confidence Level)
def calculate_prediction_interval(y, confidence_level=0.95):
    # Calculate the standard deviation of the residuals
    residuals = y - np.mean(y)
    std_dev = np.std(residuals)

    # Calculate the margin of error (z-score for 95% confidence is 1.96)
    margin_of_error = 1.96 * std_dev

    # Upper and Lower Bounds for the Confidence Interval
    y_upper = y + margin_of_error
    y_lower = y - margin_of_error
    return y_lower, y_upper


# Generate data
x, y = generate_time_series()

# Calculate the prediction interval
y_lower, y_upper = calculate_prediction_interval(y)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the time-series data
plt.plot(x, y, label='Time-Series Data', color='blue', marker='o', linestyle='-', markersize=4)

# Plot the confidence intervals
plt.fill_between(x, y_lower, y_upper, color='gray', alpha=0.5, label='95% Prediction Interval')

# Labels and title
plt.title('Time-Series with 95% Confidence Interval')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='best')

plt.grid(True)
plt.show()
