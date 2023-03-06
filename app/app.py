import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
from dash import callback

import pandas as pd
import numpy as np
import pickle
import argparse

filepath = '../models/multiple_model.pickle'

# Load the pickle file containing the trained multiple linear regression model
with open(filepath, 'rb') as file:
    model = pickle.load(file)

# Define the app and layout
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Link(
            href="https://fonts.googleapis.com/css?family=Inter:400",
            rel="stylesheet",
        ),
        html.H1(
            "Multiple Linear Regression Predictor",
            style = {'display': 'flex', 'padding-top': '15px'}
        ),
        html.Label(
            "Visits: ",
            style = {'display': 'flex'}
        ),
        dcc.Input(
            id="feature1", 
            type="number", 
            placeholder=0),
        html.Label(
            "Items: ",
            style = {'display': 'flex'}
        ),
        dcc.Input(
            id="feature2",
            type="number",
            placeholder=0
        ),
        html.Label(
            "Age: ",
            style = {'display': 'flex'}
        ),
        dcc.Input(
            id="feature3", 
            type="number", 
            placeholder=0
        ),
        html.Label(
            "Solo: ",
            style = {'display': 'flex'}
        ),
        dcc.Input(
            id="feature4",
            type="number", 
            placeholder=0
        ),
        html.Label(
            "Duplicates: ",
            style = {'display': 'flex'}
        ),
        dcc.Input(
            id="feature5",
            type="number", 
            placeholder=0
        ),
        html.Button(
            id="submit-button", 
            n_clicks=0, 
            children="Submit"
        ),
        html.Label(
            "Predicted value:",
            style = {'display': 'flex'}
        ),
        html.Div(id="output")
])

# Define the callback function to update the output
@app.callback(
    Output("output", "children"),
    [Input("submit-button", "n_clicks")],
    [State("feature1", "value"),
     State("feature2", "value"),
     State("feature3", "value"),
     State("feature4", "value"),
     State("feature5", "value")]
)
def update_output(n_clicks, feature1, feature2, feature3, feature4, feature5):
    """
    update_output is a callback function that takes in user-inputted parameters 
    and inputs them into the multiple regression model to create a prediction

    Parameters
    ----------
    n_clicks: int
        num of clicks 
    feature1: int/float
        value representing a specific feature
    feature2: int/float
        value representing a specific feature
    feature3: int/float
        value representing a specific feature
    feature4: int/float
        value representing a specific feature
    feature5: int/float
        value representing a specific feature

    Returns
    -------
    prediction: float
        predicted value
    """
    input_array = np.array([[feature1, feature2, feature3, feature4, feature5]])
    # Make the prediction using the loaded model
    print(model.predict(input_array))
    prediction = model.predict(input_array)[0]
    # Return the predicted value as the output
    return f"{prediction}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)