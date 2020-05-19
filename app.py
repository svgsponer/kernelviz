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


demo_kernel = np.array([1, 2, 1])

def add_dila_to_kernel(weights, dilation):
    
    kernel_length = len(weights) + (len(weights) - 1) * dilation
    kernel = np.zeros(kernel_length)
    kernel_length = kernel.shape[0]
    if dilation == 0:
        kernel = weights
    else:
        sup = 0
        for i in range(0, kernel_length, dilation + 1):
            kernel[i] = weights[sup]
            sup += 1
    return kernel

def apply_kernel(ts, weights, bias, dilation, padding, stride):

    # if padding > 0:
    #     _input_length = len(ts)
    #     _X = np.zeros(_input_length + (2 * padding))
    #     _X[padding:(padding + _input_length)] = ts
    #     X = _X

    # Add padding to kernel
    kernel = add_dila_to_kernel(weights, dilation)
    print(kernel)

    kernel_length = kernel.shape[0]

    input_length = ts.shape[0]
    length_diff = input_length - kernel_length
    output_length = ((length_diff) // stride) + 1
                     

    output = np.empty(output_length)

    for i in range(0, output_length):
        _sum = bias
        for j in range(0, kernel_length, stride):
            s = kernel[j] * ts[i + j]
            _sum += s
        output[i] = _sum

    print(output_length)
    print(input_length)
    print(kernel_length)
    return output


@app.callback(
    Output(component_id='ts_plot', component_property='figure'),
    [Input(component_id='ts_select', component_property='value')]
)
def plot_ts(ts_idx):
    ts_idx = int(ts_idx)
    layout = {'title': {'text':f"Time series {ts_idx}"}}
    return go.Figure(data=[go.Scatter(y=data[ts_idx, :])], layout=layout)

@app.callback(
    Output(component_id='kernel_plot', component_property='figure'),
    [Input(component_id='kernel_weights', component_property='value'),
     # Input(component_id='kernel_bias', component_property='value'),
     Input(component_id='kernel_dilation', component_property='value'),
     # Input(component_id='kernel_stride', component_property='value'),
     ]
)
def plot_kernel(kernel, dilation):
    dilation = int(dilation)
    kernel = np.array(kernel.split(','))
    dila_kernel = add_dila_to_kernel(kernel, dilation)
    layout = {'title': {'text':f'Kernel: {dila_kernel}'}}
    return go.Figure(data=[go.Scatter(y=kernel)], layout=layout)


@app.callback(
    Output(component_id='ts_trans_plot', component_property='figure'),
    [Input(component_id='ts_select', component_property='value'),
     Input(component_id='kernel_weights', component_property='value'),
     Input(component_id='kernel_bias', component_property='value'),
     Input(component_id='kernel_dilation', component_property='value'),
     Input(component_id='kernel_padding', component_property='value'),
     Input(component_id='kernel_stride', component_property='value'),
     ]
)
def plot_trans_ts(ts_idx, kernel, bias, dilatation, padding, stride):
    ts_idx = int(ts_idx)
    kernel = np.array(kernel.split(','))
    print(kernel)
    dilatation = int(dilatation)
    bias = float(bias)
    padding = int(padding)
    stride = int(stride)
    # transformed = apply_transformation(data[ts_idx,:])
    transformed = apply_kernel(data[ts_idx,:], demo_kernel, bias, dilatation, padding, stride)
    print(transformed[:5])
    print(data[ts_idx, :5])
    layout = {'title': {'text':'Transformed time series'}}
    return go.Figure(data=[go.Scatter(y=transformed)], layout=layout)


app.layout = html.Div(children=[
    html.H4(children='TS kernel visualization'),
    html.Div([
        html.Div([], className="two columns"),
        html.Div([
            html.H6(["Select time series"]),
            dcc.Slider(
                id='ts_select',
                min=-0,
                max=data.shape[1],
                step=1,
                value=0
            ),
            dcc.Graph(
                id="ts_plot"),
        ], className="ten columns"),
    ], className="row"),

    html.Div([
        html.Div([
            # Kernel values
            html.Div(children=[
                html.H5(children='Weigths'),
                dcc.Input(
                    id="kernel_weights",
                    placeholder='Enter your kernel',
                    type='text',
                    value='1,2,1',
                    debounce=True
                )]),
            # Kernel bias
            html.Div(children=[
                html.H5(children='Bias'),
                dcc.Input(
                    id="kernel_bias",
                    placeholder='Enter your bias',
                    type='number',
                    value='0',
                    debounce=True
                )]),
            # Kernel dilation
            html.Div(children=[
                html.H5(children='Dilation'),
                dcc.Input(
                    id="kernel_dilation",
                    placeholder='Enter your dilation',
                    type='number',
                    value='0',
                    debounce=True
                )]),
            # Kernel padding
            html.Div(children=[
                html.H5(children='Padding (not impl)'),
                dcc.Input(
                    id="kernel_padding",
                    placeholder='Enter your padding',
                    type='number',
                    value='0',
                    debounce=True
                )]),
            # Kernel stride
            html.Div(children=[
                html.H5(children='Stride'),
                dcc.Input(
                    id="kernel_stride",
                    placeholder='Enter your stride',
                    type='number',
                    value='1',
                    debounce=True
                )]),],className="two columns"),
        html.Div([
            dcc.Graph(
                id="kernel_plot",),],className="ten columns"),
    ], className="row"),

    dcc.Graph(
        id="ts_trans_plot")

])

if __name__ == '__main__':
    app.run_server(debug=True)
