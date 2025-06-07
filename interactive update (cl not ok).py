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


# Interactive class to handle forecasting task
class IntervalForecastingTask:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.forecasts = []  # Store forecast intervals (each forecast is a tuple: (x, lower, upper))
        self.max_forecasts = 8  # Maximum number of forecasts allowed
        self.forecast_x_positions = np.linspace(51, 58, self.max_forecasts)  # Evenly spaced x positions for forecasts
        self.click_count = 0  # To count clicks (each forecast needs two clicks)
        self.clicks_remaining = 16  # Total 16 clicks (2 for each interval)

        # Generate the original time-series data (50 observations)
        self.x, self.y = generate_time_series()

        # Calculate the 95% confidence interval (CI) for the time-series
        self.y_lower, self.y_upper = calculate_prediction_interval(self.y)

        # Plot the time-series data and confidence intervals
        self.ax.fill_between(self.x, self.y_lower, self.y_upper, color='lightblue', alpha=0.5, label="95% CI")
        self.ax.plot(self.x, self.y, label='Time-Series Data', color='blue', marker='o', linestyle='-', markersize=4)

        # Set plot limits
        self.ax.set_xlim(0, 60)  # Extend x-axis to the right of 50 (for forecasts)
        self.ax.set_ylim(0, 600)  # Adjust y-axis limits based on the data

        # Set titles and labels
        self.ax.set_title("Interval Forecasting Task")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend(loc='best')

        # Connect the mouse click event to create and modify forecast intervals
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return  # Ignore clicks outside the plot area

        # Get the clicked position (x and y coordinates)
        x_click = event.xdata
        y_click = event.ydata

        # If there are still clicks left to process
        if self.clicks_remaining > 0:
            # First click: Define the lower bound for the current forecast
            if self.click_count % 2 == 0:  # Even clicks define the lower bound
                forecast_x = self.forecast_x_positions[
                    len(self.forecasts)]  # Get the correct x position for this forecast
                self.forecasts.append((forecast_x, y_click, None))  # Store the lower bound and set upper as None
            else:  # Second click: Define the upper bound for the current forecast
                forecast_x, lower_bound, _ = self.forecasts[-1]
                self.forecasts[-1] = (forecast_x, lower_bound, y_click)  # Update the forecast with upper bound

            self.click_count += 1  # Increment the click count
            self.clicks_remaining -= 1  # Decrement clicks remaining

            # Update the plot
            self.update_plot()

    def update_plot(self):
        # Redraw the plot by clearing the axes and re-plotting everything
        self.ax.clear()

        # Plot the 95% confidence interval (CI) for the time-series data
        self.ax.fill_between(self.x, self.y_lower, self.y_upper, color='lightblue', alpha=0.5, label="95% CI")
        self.ax.plot(self.x, self.y, label='Time-Series Data', color='blue', marker='o', linestyle='-', markersize=4)

        # Plot the forecast intervals (the red intervals clicked by participants)
        for (x_click, lower, upper) in self.forecasts:
            if upper is not None:  # Only plot intervals that have both bounds
                self.ax.plot([x_click, x_click], [lower, upper], color='red',
                             lw=2)  # Red vertical line for forecast interval

        # Connect the forecast intervals with lines (optional)
        if len(self.forecasts) > 1:
            # Sort forecasts based on x-axis (time) to connect them sequentially
            sorted_forecasts = sorted(self.forecasts, key=lambda forecast: forecast[0])
            for i in range(1, len(sorted_forecasts)):
                x1, lower1, upper1 = sorted_forecasts[i - 1]
                x2, lower2, upper2 = sorted_forecasts[i]
                self.ax.plot([x1, x2], [upper1, upper2], 'r-', alpha=0.6)  # Red lines connecting upper bounds
                self.ax.plot([x1, x2], [lower1, lower2], 'r-', alpha=0.6)  # Red lines connecting lower bounds

        # Set the plot limits and labels
        self.ax.set_xlim(0, 60)  # Keep the x-axis extended for forecasts
        self.ax.set_ylim(0, 600)  # Adjust y-axis based on forecast values
        self.ax.set_title("Interval Forecasting Task")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend(loc="best")

        # Redraw the updated plot
        self.fig.canvas.draw()

    def show_plot(self):
        plt.show()


# Create an interactive interval forecasting task instance
interactive_task = IntervalForecastingTask()
interactive_task.show_plot()
