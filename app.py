# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = np.loadtxt('data/CMJ/JumpResampled/JumpResampled_TRAIN', delimiter=',')[:,1:]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


@app.callback(
    Output(component_id='ts_plot', component_property='figure'),
    [Input(component_id='ts_select', component_property='value')]
)
def plot_ts(ts_idx):
    ts_idx = int(ts_idx)
    return go.Figure(data=[go.Scatter(y=data[ts_idx, :])])


app.layout = html.Div(children=[
    html.H4(children='Data'),
    dcc.Slider(
        id='ts_select',
        min=-0,
        max=data.shape[1],
        step=1,
        value=0
    ),
    dcc.Graph(
        id="ts_plot")

])

if __name__ == '__main__':
    app.run_server(debug=True)
