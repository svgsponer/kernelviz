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

datafile='data/CMJ/JumpResampled/JumpResampled_TRAIN'
#datafile='../data/1M-TSC-SITS_2006_NDVI_C/SITS1M_fold1/SITS1M_fold1_TRAIN.csv'
data = pd.DataFrame(np.loadtxt(datafile, delimiter=',')[:, 1:].T)
print(data.shape)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

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
            print(kernel[j])
            s = kernel[j] * ts[i + j]
            print(s)
            _sum += s
            output[i] = _sum

    print(output_length)
    print(input_length)
    print(kernel_length)
    print(output)
    return output


def noramlize(ts):
    ts = (ts-ts.mean())/ts.std()
    return ts


@app.callback(
    Output(component_id='current-TS', component_property='children'),
    [Input(component_id='ts_select', component_property='value'),
     Input(component_id='ts-chkb', component_property='value'),
     Input(component_id='test-ts', component_property='value'),
     ]
)
def prepare_ts(ts_idx, chk_list, test_ts):
    if 'test' in chk_list:
        ts = np.array(list(map(float, test_ts.split(','))))
        ts = pd.DataFrame(ts)
    else:
        ts_idx = int(ts_idx)
        ts = data[ts_idx]

    if 'norm' in chk_list:
        ts = noramlize(ts)

    return ts.to_json(orient='values')


@app.callback(
    Output(component_id='ts_plot', component_property='figure'),
    [Input(component_id='current-TS', component_property='children')]
)
def plot_ts(json_data):
    dff = pd.read_json(json_data, orient='values').to_numpy().flatten()
    print(dff)
    layout = {'title': {'text': f"Time series"}}
    return go.Figure(data=[go.Scatter(y=dff)])


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
    [Input(component_id='current-TS', component_property='children'),
     Input(component_id='kernel_weights', component_property='value'),
     Input(component_id='kernel_bias', component_property='value'),
     Input(component_id='kernel_dilation', component_property='value'),
     Input(component_id='kernel_padding', component_property='value'),
     Input(component_id='kernel_stride', component_property='value'),
     ]
)
def plot_trans_ts(json_data, kernel, bias, dilatation, padding, stride):

    dff = pd.read_json(json_data, orient='values').to_numpy().flatten()
    kernel = np.array(list(map(float, kernel.split(','))))
    print(kernel)
    dilatation = int(dilatation)
    bias = float(bias)
    padding = int(padding)
    stride = int(stride)
    # transformed = apply_transformation(data[ts_idx,:])
    transformed = apply_kernel(dff, kernel, bias, dilatation, padding, stride)
    layout = {'title': {'text':'Transformed time series'}}
    return go.Figure(data=[go.Scatter(y=transformed)], layout=layout)


app.layout = html.Div(children=[
    html.H4(children='TS kernel visualization'),
    # Div to store the current TS
    html.Div(id='current-TS', style={'display': 'none'}),

    # Time series plot
    html.Div([
        html.Div([
            html.Div(children=[
                dcc.Checklist(
                    options=[
                        {'label': 'Normalize', 'value': 'norm'},
                        {'label': 'Test sequence', 'value': 'test'},
                    ],
                    value=[],
                    id='ts-chkb'
                )]),

            # Test time series
            html.Div(children=[
                html.H5(children='Test time serie'),
                dcc.Input(
                    id="test-ts",
                    placeholder='Enter your test time series',
                    type='text',
                    value='1,1,1,1,1,1,1,1,1,1,1,1,1',
                    debounce=True
                )])],
            className="two columns"),
        html.Div([
            html.H6(["Select time series"]),
            dcc.Slider(
                id='ts_select',
                min=-0,
                max=data.shape[0],
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
                html.H5(children='Padding (not impl.)'),
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

    html.Div([
        html.Div(["t "], className="two columns"),
        html.Div([
            dcc.Graph(
                id="ts_trans_plot")
        ], className="ten columns"),
    ], className="row"),
])

if __name__ == '__main__':
    app.run_server(debug=True)
