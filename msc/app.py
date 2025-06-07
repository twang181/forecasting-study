import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create a simple layout
app.layout = dbc.Container([
    html.H1("Welcome to Your Dash App", className="text-center mt-4"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H4("Test Component"),
            html.P("If you can see this page with Bootstrap styling, your setup is working correctly!"),
            dbc.Button("Test Button", id="test-button", color="primary", className="mb-3"),
            html.Div(id="output-message")
        ], width={"size": 6, "offset": 3})
    ])
], fluid=True)

@app.callback(
    Output("output-message", "children"),
    Input("test-button", "n_clicks"),
    prevent_initial_call=True
)
def update_output(n_clicks):
    if n_clicks:
        return dbc.Alert(
            f"Button was clicked {n_clicks} times!",
            color="success",
            className="mt-3"
        )

if __name__ == '__main__':
    app.run(debug=True)