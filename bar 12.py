import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import csv
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# Read the time-series data from the CSV file
simulated_df = pd.read_csv('simulated_series_new.csv')
forecast_df = pd.read_csv('forecast_table.csv')

# Number of time series (columns) in the dataframe
n_series = simulated_df.shape[1] - 1  # Subtract 1 for the 'Time' column

# Experimental conditions
conditions = ["control", "80_point_interval", "95_point_interval", "80_interval", "95_interval"]


def assign_condition(participant_id):
    try:
        df = pd.read_csv("participant_assignments.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["participant_id", "condition"])

    counts = df["condition"].value_counts().reindex(conditions, fill_value=0)
    total = counts.sum() + 1

    probs = 1 - (counts / total)
    probs = probs / probs.sum()

    condition = np.random.choice(conditions, p=probs)

    new_row = pd.DataFrame([{"participant_id": participant_id, "condition": condition}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("participant_assignments.csv", index=False)

    return condition


class InteractiveForecast:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.participant_id = str(uuid.uuid4())
        self.condition = assign_condition(self.participant_id)
        self.forecasts = forecast_df
        self.max_forecasts = 4
        self.forecast_x_positions = np.linspace(49, 52, self.max_forecasts)

        print(f"Assigned Condition: {self.condition}")

        self.selected_series = np.random.choice(range(1, n_series + 1), size=24, replace=False)
        self.series_counter = 0
        self.series_index = self.selected_series[self.series_counter]

        self.x = simulated_df['time'].values
        self.y_noisy = simulated_df[f"series_{self.series_index}"].values

        self.user_intervals = {}
        self.setup_layout()
        self.setup_callbacks()

    def create_figure(self):
        fig = go.Figure()

        # Plot actual noisy data points
        fig.add_trace(go.Scatter(
            x=self.x,
            y=self.y_noisy,
            mode='lines+markers',
            name=f'Observed Sales (Series{self.series_index})',
            line=dict(color='black', width=1),
            marker=dict(size=5)
        ))

        # Add forecast intervals if not in control condition
        if self.condition != "control":
            confidence_level = "95" if "95" in self.condition else "80"
            lower_col = f"lower_{confidence_level}"
            upper_col = f"upper_{confidence_level}"

            for x_pos in self.forecast_x_positions:
                row = self.forecasts[
                    (self.forecasts['Series'] == self.series_index) &
                    (self.forecasts['time'] == x_pos)
                    ]

                if not row.empty:
                    lower = row[lower_col].values[0]
                    upper = row[upper_col].values[0]

                    # Add draggable interval bar
                    fig.add_shape(
                        type="line",
                        x0=x_pos,
                        x1=x_pos,
                        y0=lower,
                        y1=upper,
                        line=dict(
                            color="rgba(135, 206, 235, 0.4)",
                            width=20,
                        ),
                        name=f'Interval_{x_pos}',
                        editable=True
                    )

                    if "point" in self.condition:
                        point_forecast = np.mean([lower, upper])
                        fig.add_trace(go.Scatter(
                            x=[x_pos],
                            y=[point_forecast],
                            mode='markers',
                            marker=dict(size=8, color='blue'),
                            name='Point Forecast' if x_pos == self.forecast_x_positions[0] else None,
                            showlegend=bool(x_pos == self.forecast_x_positions[0])
                        ))

        # Update layout
        y_min, y_max = np.min(self.y_noisy), np.max(self.y_noisy)
        margin = (y_max - y_min) * 0.2

        fig.update_layout(
            title=f"Time Series: Series{self.series_index}",
            xaxis=dict(
                title="Sales Period",
                range=[0, 53],
                tickmode='array',
                tickvals=list(range(5, 55, 5)),
                ticktext=[str(i) for i in range(5, 55, 5)],
                fixedrange=True  # Prevent x-axis movement
            ),
            yaxis=dict(
                title="Number of Sales",
                range=[y_min - margin, y_max + margin],
                fixedrange=True  # Prevent y-axis movement
            ),
            showlegend=True,
            hovermode='closest',
            dragmode=None  # Disable background dragging
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

        return fig

    def setup_layout(self):
        # Configure mode bar buttons based on condition
        mode_bar_buttons = ['drawline', 'eraseshape'] if self.condition == "control" else []

        self.app.layout = dbc.Container([
            dcc.Store(id='user-intervals', data={}),
            dcc.Graph(
                id='forecast-plot',
                figure=self.create_figure(),
                config={
                    'scrollZoom': False,
                    'displayModeBar': True,
                    'editable': True,
                    'modeBarButtonsToAdd': mode_bar_buttons,
                    'edits': {
                        'shapePosition': True,
                        'legendPosition': False,
                        'titleText': False
                    }
                }
            ),
            dbc.Button("Save", id="save-button", color="primary", className="mt-2"),
            html.Div(id='save-status')
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('forecast-plot', 'figure'),
             Output('save-status', 'children'),
             Output('user-intervals', 'data')],
            [Input('save-button', 'n_clicks'),
             Input('forecast-plot', 'relayoutData')],
            [State('user-intervals', 'data'),
             State('forecast-plot', 'figure')],
            prevent_initial_call=True
        )
        def update_plot(n_clicks, relayout_data, user_intervals, current_fig):
            triggered_id = ctx.triggered_id

            if triggered_id == 'save-button' and n_clicks:
                self.save_data(user_intervals)
                self.switch_to_new_series()
                return self.create_figure(), "Data saved successfully!", {}

            elif triggered_id == 'forecast-plot' and relayout_data:
                # Handle shape dragging/resizing
                if any(key.startswith('shapes') for key in relayout_data.keys()):
                    for key, value in relayout_data.items():
                        if key.startswith('shapes['):
                            shape_idx = int(key.split('[')[1].split(']')[0])
                            shape = current_fig['layout']['shapes'][shape_idx]

                            # Ensure x position stays fixed
                            if 'x0' in shape:
                                shape['x1'] = shape['x0']
                            elif 'x1' in shape:
                                shape['x0'] = shape['x1']

                            x_pos = shape['x0']  # x0 and x1 are the same

                            # Store the updated values
                            if str(x_pos) not in user_intervals:
                                user_intervals[str(x_pos)] = {}
                            user_intervals[str(x_pos)]['lower'] = min(shape['y0'], shape['y1'])
                            user_intervals[str(x_pos)]['upper'] = max(shape['y0'], shape['y1'])

                return current_fig, "", user_intervals

            return current_fig, "", user_intervals

    def save_data(self, user_intervals):
        forecast_results = []

        if self.condition == "control":
            for x_pos, interval in user_intervals.items():
                forecast_results.append((float(x_pos), interval['lower'], interval['upper']))
        else:
            for x_pos in self.forecast_x_positions:
                row = self.forecasts[
                    (self.forecasts['Series'] == self.series_index) &
                    (self.forecasts['time'] == x_pos)
                    ]
                if not row.empty:
                    confidence_level = "95" if "95" in self.condition else "80"
                    lower = row[f"lower_{confidence_level}"].values[0]
                    upper = row[f"upper_{confidence_level}"].values[0]
                    forecast_results.append((x_pos, lower, upper))

        with open('forecast_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Participant_ID", "Series", "Period", "Lower", "Upper", "Condition"])
            for period, lower, upper in forecast_results:
                writer.writerow([self.participant_id, self.series_index, period, lower, upper, self.condition])

    def switch_to_new_series(self):
        self.series_counter += 1
        if self.series_counter >= len(self.selected_series):
            print("All 24 series completed.")
            return

        self.series_index = self.selected_series[self.series_counter]
        self.y_noisy = simulated_df[f"series_{self.series_index}"].values
        self.user_intervals = {}

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    interactive_forecast = InteractiveForecast()
    interactive_forecast.run()