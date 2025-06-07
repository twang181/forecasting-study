import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import csv
import dash #add
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


        #revised:
        # Step 1: Define 12 pairs of series (based on your data design)
        type_pairs = [(i, i + 1) for i in range(1, 24, 2)]  # [(1,2), (3,4), ..., (23,24)]

        # Step 2: Randomly assign one in each pair to positive (1) and the other to negative (0)
        selected_series = []
        nudge_flags = []

        for pair in type_pairs:
            pos_index, neg_index = np.random.permutation(pair)
            selected_series.extend([pos_index, neg_index])
            nudge_flags.extend([1, 0])  # 1 = positive nudge, 0 = negative nudge

        # Step 3: Shuffle the 24 trials randomly
        combined = list(zip(selected_series, nudge_flags))
        np.random.shuffle(combined)

        self.selected_series, self.nudge_assignments = zip(*combined)
        self.selected_series = list(self.selected_series)
        self.nudge_assignments = list(self.nudge_assignments)

        # Initialize first trial
        self.series_counter = 0
        self.series_index = self.selected_series[self.series_counter]
        self.current_nudge = self.nudge_assignments[self.series_counter]

       # self.series_counter = 0
       # self.series_index = self.selected_series[self.series_counter]

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
                            color="rgba(100, 149, 237, 0.6)",
                            width=10,
                        ),
                        editable=True,
                        layer='above'
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
        margin = (y_max - y_min) * 0.5  # Increased margin for more dragging room

        fig.update_layout(
            #title=f"Time Series: Series{self.series_index}",
            xaxis=dict(
                title="Sales Period",
                range=[0, 53],
                tickmode='array',
                tickvals=list(range(5, 55, 5)),
                ticktext=[str(i) for i in range(5, 55, 5)],
                fixedrange=True,
                constrain='domain'
            ),
            yaxis=dict(
                title="Number of Sales",
                range=[y_min - margin, y_max + margin],
                fixedrange=True
            ),
            showlegend=True,
            hovermode='closest',
            dragmode=False  # No background dragging
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

        return fig

    def setup_layout(self):
        self.app.layout = dbc.Container([
            dcc.Store(id='user-intervals', data={}),
            dcc.Store(id='click-buffer', data={'points': []}),

            # NEW: Add this for displaying nudge messages
            # html.Div(id='nudge-message', style={'marginTop': '10px', 'fontWeight': 'bold'}),
            html.Div(
                id='nudge-message',
                children=self.get_nudge_text(),  # 👈 Add initial value
                style={
                    'marginTop': '20px',  # Space between plot and nudge
                    'fontWeight': 'bold',
                    'fontSize': '18px',
                    'color': '#004085'  # Optional: a shade of blue
                }
            ),

            dcc.Graph(
                id='forecast-plot',
                figure=self.create_figure(),
                config={
                    'scrollZoom': False,
                    'displayModeBar': True,
                    'editable': True,

                    'edits': {
                        'shapePosition': True,
                        'titleText': False  # ⬅️ explicitly disable title editing
                    },

                    'modeBarButtonsToAdd': ['drawline', 'eraseshape'] if self.condition == "control" else [],
                    'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'lasso2d'],
                    #'edits': {'shapePosition': True},
                    'displaylogo': False,
                    'staticPlot': False
                },
                clickmode='event+select',
                style={'height': '80vh'},
                #clickData=None
            ),

            dbc.Button("Save", id="save-button", color="primary", className="mt-2"),
            html.Div(id='save-status')
        ])

    #add
    def get_nudge_text(self):
        if self.current_nudge == 1:
            return "This product is currently promoted and may have a bright future."
        else:
            return "This product has not been promoted and may have a damp future."

    def setup_callbacks(self):
        @self.app.callback(
            [Output('forecast-plot', 'figure'),
             Output('save-status', 'children'),
             Output('user-intervals', 'data'),
             Output('click-buffer', 'data'),
             Output('nudge-message', 'children')], #newly add
            [Input('save-button', 'n_clicks'),
             Input('forecast-plot', 'relayoutData'),
             Input('forecast-plot', 'clickData')],
            [State('user-intervals', 'data'),
             State('forecast-plot', 'figure'),
             State('click-buffer', 'data')],
            prevent_initial_call=True
        )

        def update_plot(n_clicks, relayout_data, click_data, user_intervals, current_fig, click_buffer):
            triggered_id = ctx.triggered_id

            if triggered_id == 'save-button' and n_clicks:
                self.save_data(user_intervals)
                self.switch_to_new_series()

                # 👇 Create message based on nudge
                if self.current_nudge == 1:
                    nudge_message = "This product is currently promoted and expected to perform better than usual."
                else:
                    nudge_message = "This product has not been promoted recently and may receive less attention."

                return self.create_figure(), "Data saved successfully!", {}, {'points': []}, nudge_message
                #return self.create_figure(), "Data saved successfully!", {}, {'points': []}

            elif triggered_id == 'forecast-plot' and relayout_data:
                if any(key.startswith('shapes[') for key in relayout_data.keys()):
                    y_axis_range = current_fig['layout']['yaxis']['range']
                    y_min, y_max = y_axis_range[0], y_axis_range[1]

                    for key, value in relayout_data.items():
                        if key.startswith('shapes['):
                            shape_idx = int(key.split('[')[1].split(']')[0])

                            try:
                                shape = current_fig['layout']['shapes'][shape_idx]
                            except IndexError:
                                continue

                            if self.condition == "control":
                                x0 = shape['x0']
                                x1 = shape['x1']
                                fixed_x = round(np.mean([x0, x1]), 2)
                                shape['x0'] = shape['x1'] = fixed_x

                                y0_key = f'shapes[{shape_idx}].y0'
                                y1_key = f'shapes[{shape_idx}].y1'

                                if y0_key in relayout_data:
                                    shape['y0'] = max(min(relayout_data[y0_key], y_max), y_min)
                                if y1_key in relayout_data:
                                    shape['y1'] = max(min(relayout_data[y1_key], y_max), y_min)

                                y_low = min(shape['y0'], shape['y1'])
                                y_high = max(shape['y0'], shape['y1'])

                                user_intervals[str(fixed_x)] = {
                                    'lower': y_low,
                                    'upper': y_high
                                }

                            else:
                                if shape_idx < len(self.forecast_x_positions):
                                    original_x = self.forecast_x_positions[shape_idx]
                                    shape['x0'] = shape['x1'] = original_x

                                    y0_key = f'shapes[{shape_idx}].y0'
                                    y1_key = f'shapes[{shape_idx}].y1'

                                    if y0_key in relayout_data:
                                        shape['y0'] = max(min(relayout_data[y0_key], y_max), y_min)
                                    if y1_key in relayout_data:
                                        shape['y1'] = max(min(relayout_data[y1_key], y_max), y_min)

                                    y_low = min(shape['y0'], shape['y1'])
                                    y_high = max(shape['y0'], shape['y1'])

                                    user_intervals[str(original_x)] = {
                                        'lower': y_low,
                                        'upper': y_high
                                    }

                #return current_fig, "", user_intervals, click_buffer
                return current_fig, "", user_intervals, click_buffer, dash.no_update

                print("Click Data:", click_data)

            elif triggered_id == 'forecast-plot' and click_data and self.condition == "control":
                print("Click Data:", click_data)  # <-- ADD THIS
                x_clicked = click_data['points'][0]['x']
                y_clicked = click_data['points'][0]['y']
                click_buffer['points'].append((x_clicked, y_clicked))

                if len(click_buffer['points']) == 2:
                    x_avg = round(np.mean([p[0] for p in click_buffer['points']]), 2)
                    y_vals = [p[1] for p in click_buffer['points']]
                    y_low = min(y_vals)
                    y_high = max(y_vals)

                    current_fig['layout'].setdefault('shapes', []).append({
                        'type': 'line',
                        'x0': x_avg,
                        'x1': x_avg,
                        'y0': y_low,
                        'y1': y_high,
                        'line': {
                            'color': 'rgba(100, 149, 237, 0.6)',
                            'width': 10
                        }
                    })

                    user_intervals[str(x_avg)] = {
                        'lower': y_low,
                        'upper': y_high
                    }

                    click_buffer['points'] = []

                #return current_fig, "", user_intervals, click_buffer
                return current_fig, "", user_intervals, click_buffer, dash.no_update

            #return current_fig, "", user_intervals, click_buffer
            return current_fig, "", user_intervals, click_buffer, dash.no_update

    def save_data(self, user_intervals):
        forecast_results = []

        if self.condition == "control":
            for x_pos, interval in user_intervals.items():
                forecast_results.append((x_pos, interval['lower'], interval['upper']))
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
        self.current_nudge = self.nudge_assignments[self.series_counter]  # 👈 ADD THIS LINE
        self.y_noisy = simulated_df[f"series_{self.series_index}"].values
        self.user_intervals = {}

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    interactive_forecast = InteractiveForecast()
    interactive_forecast.run()

    from dash import callback_context


    @interactive_forecast.app.callback(
        Output('save-status', 'children'),
        Input('forecast-plot', 'clickData')
    )
    def debug_click(clickData):
        print("CLICK DEBUG:", clickData)
        if clickData:
            return f"You clicked: {clickData['points'][0]['x']}, {clickData['points'][0]['y']}"
        return dash.no_update
