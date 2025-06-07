import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv  # To save the data into a CSV file

# Generate Time-Series Data
def generate_time_series():
    x = np.arange(1, 51)  # Time steps from 1 to 50
    k = 1.5  # Power-law exponent
    y = 50 + 300 * (x / 50) ** k  # Power-law function
    noise = np.random.normal(0, 3, len(x))  # Add Gaussian noise
    return x, y + noise


# Calculate Prediction Interval (95% Confidence Level) for given y-values
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
        self.fig, self.ax = plt.subplots(figsize=(8, 5))  # Made the plot less wide
        self.forecasts = []  # Store forecast intervals (each forecast is a tuple: (x, lower, upper))
        self.max_forecasts = 8  # Maximum number of forecasts allowed
        self.forecast_x_positions = np.linspace(51, 58, self.max_forecasts)  # Evenly spaced x positions for forecasts
        self.click_count = 0  # To count clicks (each forecast needs two clicks)
        self.clicks_remaining = 16  # Total 16 clicks (2 for each interval)

        # Store clicked points (to save later)
        self.clicked_points = []  # Store the clicked points: (forecast_x, lower, upper)

        # Generate the original time-series data (50 observations)
        self.x, self.y = generate_time_series()

        # Calculate the 95% prediction interval for the first 50 points
        self.y_lower, self.y_upper = calculate_prediction_interval(self.y)

        # Plot the 95% CI for the first 50 points as small points (no line)
        self.ax.plot(self.x, self.y_lower, 'ro', label='Lower Bound CI (95%)', markersize=4)  # Red dots for lower bound
        self.ax.plot(self.x, self.y_upper, 'go', label='Upper Bound CI (95%)',
                     markersize=4)  # Green dots for upper bound

        # Set plot limits (rescale y-axis to fit the data better)
        self.ax.set_xlim(0, 60)  # Extend x-axis to the right of 50 (for forecasts)
        self.ax.set_ylim(min(self.y_lower) - 10, max(self.y_upper) + 10)  # Narrow the y-axis range visually

        # Set titles and labels
        self.ax.set_title("Interval Forecasting Task")
        self.ax.set_xlabel("Sales Period")  # Set x-axis label
        self.ax.set_ylabel("Number of Sales")  # Set y-axis label

        # Hide the y-axis values (tick labels)
        self.ax.set_yticklabels([])

        # Set custom x-axis ticks
        self.ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])  # Custom x-ticks
        self.ax.set_xticklabels([f"{i}" for i in range(5, 61, 5)])  # Display numbers 5, 10, 15, ...

        # Add the legend for the axes (now showing the legend for the CI bounds)
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

                # Save the clicked points to the list for later
                self.clicked_points.append((forecast_x, lower_bound, y_click))

            self.click_count += 1  # Increment the click count
            self.clicks_remaining -= 1  # Decrement clicks remaining

            # Update the plot
            self.update_plot()

    def update_plot(self):
        # Redraw only the newly clicked points, without clearing previous points
        # Plot the 95% CI for the first 50 points as small points (no line)
        self.ax.plot(self.x, self.y_lower, 'ro', label='Lower Bound CI (95%)', markersize=4)  # Red dots for lower bound
        self.ax.plot(self.x, self.y_upper, 'go', label='Upper Bound CI (95%)',
                     markersize=4)  # Green dots for upper bound

        # Plot the forecast intervals (the small points clicked by participants)
        for (x_click, lower, upper) in self.forecasts:
            if upper is not None:  # Only plot intervals that have both bounds
                self.ax.plot(x_click, lower, 'bo')  # Small blue dot for lower bound (after click)
                self.ax.plot(x_click, upper, 'bo')  # Small blue dot for upper bound (after click)

                # Fill between the bounds to visualize the interval
                self.ax.fill_between([x_click, x_click], lower, upper, color='blue', alpha=0.3)

        # Optionally, extend the confidence region if forecasts progress (expand CI area)
        if len(self.forecasts) > 0:
            # Extend the CI area for each forecast made
            extended_x = np.concatenate([self.x, self.forecast_x_positions[:len(self.forecasts)]])
            self.ax.set_xlim(0, 60)  # Keep the x-axis extended for forecasts
            self.ax.set_ylim(min(self.y_lower) - 10, max(self.y_upper) + 10)  # Adjust y-axis for CI and forecasts

        # Set titles and labels
        self.ax.set_title("Interval Forecasting Task")
        self.ax.set_xlabel("Sales Period")
        self.ax.set_ylabel("Number of Sales")

        # Hide the y-axis values (tick labels)
        self.ax.set_yticklabels([])

        # Set custom x-axis ticks
        self.ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])  # Custom x-ticks
        self.ax.set_xticklabels([f"{i}" for i in range(5, 61, 5)])  # Display numbers 5, 10, 15, ...

        # Redraw the updated plot
        self.fig.canvas.draw()

    def save_clicked_data(self, filename="forecast_data.csv"):
        # Save the clicked data (forecast intervals) to a CSV file
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Forecast Period', 'First click', 'Secind click'])  # CSV header
            for data in self.clicked_points:
                writer.writerow(data)  # Write each forecast's lower and upper bounds

        print(f"Data saved to {filename}")

    def show_plot(self):
        plt.show()


# Create an interactive interval forecasting task instance
interactive_task = IntervalForecastingTask()
interactive_task.show_plot()

# Save the data after the interactive session ends (you can manually call this after the task is done)
interactive_task.save_clicked_data()  # Call this when you want to save the data
