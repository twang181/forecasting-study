import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import norm


# Generate Time-Series Data
def generate_time_series():
    x = np.arange(1, 51)  # Time steps from 1 to 50
    k = 1.5  # Power-law exponent
    y = 50 + 300 * (x / 50) ** k  # Power-law function
    noise = np.random.normal(0, 3, len(x))  # Add Gaussian noise
    return x, y + noise


# Calculate Prediction Interval (95% Confidence Level) for given y-values
def calculate_prediction_interval(y, confidence_level=0.95):
    residuals = y - np.mean(y)
    std_dev = np.std(residuals)
    margin_of_error = 1.96 * std_dev
    y_upper = y + margin_of_error
    y_lower = y - margin_of_error
    return y_lower, y_upper


class InteractiveForecast:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.forecasts = []  # Store forecast intervals as tuples (x, lower, upper)
        self.max_forecasts = 8
        self.forecast_x_positions = np.linspace(51, 58, self.max_forecasts)

        # Generate the time-series data
        self.x, self.y = generate_time_series()
        self.y_lower, self.y_upper = calculate_prediction_interval(self.y)

        # Plot the original confidence bounds
        self.ax.plot(self.x, self.y_lower, 'ro', markersize=4, label="Lower Bound (95%)")
        self.ax.plot(self.x, self.y_upper, 'go', markersize=4, label="Upper Bound (95%)")

        # Y-axis scaling
        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(min(self.y_lower) - 20, max(self.y_upper) + 20)

        # Axis Labels
        self.ax.set_xlabel("Sales Period")
        self.ax.set_ylabel("Number of Sales")
        self.ax.set_yticklabels([])
        self.ax.set_xticks(np.arange(5, 61, 5))
        self.ax.set_xticklabels([str(i) for i in range(5, 61, 5)])
        self.ax.legend()

        # Initialize draggable forecast points at 95% CL
        self.draggable_points = []
        last_lower, last_upper = self.y_lower[-1], self.y_upper[-1]  # Use last point's CL as starting forecast

        for i in range(len(self.forecast_x_positions)):
            x_pos = self.forecast_x_positions[i]
            lower_point, = self.ax.plot(x_pos, last_lower, 'bo', markersize=6, picker=True)
            upper_point, = self.ax.plot(x_pos, last_upper, 'bo', markersize=6, picker=True)
            self.draggable_points.append((x_pos, lower_point, upper_point))

        # Connect Events
        self.selected_point = None
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Save Button
        self.save_button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.save_button = Button(self.save_button_ax, "Save")
        self.save_button.on_clicked(self.save_data)

    def on_pick(self, event):
        """Selects a point when clicked."""
        for x_pos, lower_point, upper_point in self.draggable_points:
            if event.artist == lower_point:
                self.selected_point = (x_pos, "lower", lower_point)
                break
            elif event.artist == upper_point:
                self.selected_point = (x_pos, "upper", upper_point)
                break

    def on_drag(self, event):
        """Moves the selected point while dragging."""
        if self.selected_point and event.inaxes == self.ax:
            x_pos, bound_type, point = self.selected_point
            point.set_ydata([event.ydata])  # ✅ Convert ydata to a sequence
            self.fig.canvas.draw()

    def on_release(self, event):
        """Stops moving the point when mouse is released."""
        self.selected_point = None  # ✅ Stop dragging when mouse is released

    def save_data(self, event):
        """Saves forecast intervals."""
        forecast_results = []
        for x_pos, lower_point, upper_point in self.draggable_points:
            lower_y = lower_point.get_ydata()[0]
            upper_y = upper_point.get_ydata()[0]
            forecast_results.append((x_pos, lower_y, upper_y))

        print("Saved Forecasts:", forecast_results)  # You can store this in a file or database

    def show_plot(self):
        plt.show()


# Run the Interactive Forecasting Tool
interactive_forecast = InteractiveForecast()
interactive_forecast.show_plot()
