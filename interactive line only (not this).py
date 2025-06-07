import numpy as np
import matplotlib.pyplot as plt


class InteractiveForecastingTask:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.forecasts = []  # Store forecast intervals (each forecast is a tuple: (x, y))
        self.max_forecasts = 8  # Maximum number of forecasts allowed
        self.forecast_x_positions = np.linspace(51, 58, self.max_forecasts)  # Evenly spaced x positions for forecasts

        # Generate the original time-series data (50 observations)
        self.x = np.arange(1, 51)  # Time steps from 1 to 50
        self.y = 50 + 300 * (self.x / 50) ** 1.5  # Power-law function (no noise for simplicity)

        # Plot the time-series data (the first 50 observations)
        self.ax.plot(self.x, self.y, label="Time-Series Data", color='blue', marker='o', linestyle='-', markersize=4)
        self.ax.set_xlim(0, 60)  # Extend x-axis to the right of 50 (for forecasts)
        self.ax.set_ylim(0, 600)  # Adjust y-axis limits based on the data

        self.ax.set_title("Interactive Forecasting Task")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend(loc="best")

        # Connect the mouse click event to create and modify forecasts
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return  # Ignore clicks outside the plot area

        # Get the clicked position (x and y coordinates)
        x_click = event.xdata
        y_click = event.ydata

        # Check if forecast points are allowed and if the click is within the forecast range
        if len(self.forecasts) < self.max_forecasts:
            # Place forecast at a fixed x position based on the index in self.forecasts
            forecast_x = self.forecast_x_positions[len(self.forecasts)]

            # Add the forecast as a (x, y) pair
            self.forecasts.append((forecast_x, y_click))

            # Update the plot
            self.update_plot()

    def update_plot(self):
        # Redraw the plot by clearing the axes and re-plotting everything
        self.ax.clear()

        # Re-plot the original time-series data
        self.ax.plot(self.x, self.y, label="Time-Series Data", color='blue', marker='o', linestyle='-', markersize=4)

        # Plot the forecast intervals (the red points clicked by participants)
        for (x_click, y_click) in self.forecasts:
            self.ax.plot(x_click, y_click, 'ro')  # Red points for forecasts

        # Connect the forecast intervals with lines
        if len(self.forecasts) > 1:
            # Sort forecasts based on x-axis (time) to connect them sequentially
            sorted_forecasts = sorted(self.forecasts, key=lambda forecast: forecast[0])
            for i in range(1, len(sorted_forecasts)):
                x1, y1 = sorted_forecasts[i - 1]
                x2, y2 = sorted_forecasts[i]
                self.ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.6)  # Red lines connecting forecasts

        # Set the plot limits and labels
        self.ax.set_xlim(0, 60)  # Keep the x-axis extended for forecasts
        self.ax.set_ylim(0, 600)  # Adjust y-axis based on forecast values
        self.ax.set_title("Interactive Forecasting Task")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend(loc="best")

        # Redraw the updated plot
        self.fig.canvas.draw()

    def show_plot(self):
        plt.show()


# Create an interactive forecasting task instance
interactive_task = InteractiveForecastingTask()
interactive_task.show_plot()
