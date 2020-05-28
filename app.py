# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import json
import base64
import io
import utils as utls

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "KernelViz"
demo_kernel = np.array([1, 2, 1])


@app.callback(
    [Output(component_id='test-ts-container', component_property='style'),
     Output(component_id='ts-select-container', component_property='style'),],
    [Input(component_id='test_ts_switch', component_property='on'),],)

def update_test_ts_style(use_test_ts):
    if not use_test_ts:
        test_ts_style = {'display': 'none'}
        slider_style = {}
    else:
        test_ts_style = {}
        slider_style = {'display': 'none'}
    return test_ts_style, slider_style


@app.callback(
    dash.dependencies.Output('ts-slider-output-container', 'children'),
    [dash.dependencies.Input('ts-select', 'value')])
def update_output_ts_input(value):
    return "{}".format(value)


# Could be split as a change in normalize should not need to reread the full
# dataset but only the currently selected ts
@app.callback(
    Output(component_id='current-TS', component_property='children'),
    [Input(component_id='ts-select', component_property='value'),
     Input(component_id='normalize_ts_switch', component_property='on'),
     Input(component_id='test_ts_switch', component_property='on'),
     Input(component_id='test-ts', component_property='value'),
     Input(component_id='all-data', component_property='children'),
     ]
)
def prepare_ts(ts_idx, do_normalize, do_use_test_ts, test_ts, data_json):
    print('Prepare TS')
    if do_use_test_ts:
        ts = np.array(list(map(float, test_ts.split(','))))
        ts = pd.DataFrame(ts)
    else:
        data = pd.read_json(data_json, orient='values')
        ts_idx = int(ts_idx)
        ts = data[ts_idx]

    if do_normalize:
        ts = utls.noramlize(ts)
    return ts.to_json(orient='values')


@app.callback(
    Output(component_id='ts_plot', component_property='figure'),
    [Input(component_id='current-TS', component_property='children')]
)
def plot_ts(json_data):
    print('Plot TS')
    dff = pd.read_json(json_data, orient='values').to_numpy().flatten()
    layout = {'title': {'text': "Time series"}}
    return go.Figure(data=[go.Scatter(y=dff)], layout=layout)


@app.callback(
    Output(component_id='current-kernel', component_property='children'),
    [Input(component_id='kernel_weights', component_property='value'),
     Input(component_id='kernel_centering_chkb', component_property='on'),
     Input(component_id='kernel_dilation', component_property='value'),
     ])
def prepare_kernel(kernel, kc_ckhkb, dilation):
    dilation = int(dilation)
    kernel = np.array(list(map(float, kernel.split(','))))
    if kc_ckhkb:
        mean = np.mean(kernel)
        kernel = kernel - mean
    dila_kernel = utls.add_dila_to_kernel(kernel, dilation)
    ret = {'original_kernel': kernel.tolist(),
           'dila_kernel': dila_kernel.tolist()}
    return json.dumps(ret)


@app.callback(
    Output(component_id='kernel_plot', component_property='figure'),
    [Input(component_id='current-kernel', component_property='children'),
     Input(component_id='current-TS', component_property='children'),
     Input(component_id='kernel-plot-with-dila', component_property='on'),]
)
def plot_kernel(kernel_json, ts_json, plot_with_dila):
    bar_scale = 0.01
    margin = 0.1
    kernel_json = json.loads(kernel_json)
    kernel = np.array(kernel_json['original_kernel'])
    dila_kernel = np.array(kernel_json['dila_kernel'])

    ts_length = pd.read_json(ts_json, orient='values').to_numpy().flatten().shape[0]

    layout = {'title': {'text': f'Kernel: {kernel}\n Dilated kernel: {dila_kernel}'},
              'xaxis': {'range': [-1, kernel.shape[0]+1]}}
    if plot_with_dila:
        layout = {'title': {'text': f'Dilated kernel: {dila_kernel}'},
                  'xaxis': {'range':
                            [-dila_kernel.shape[0]*margin,
                             (dila_kernel.shape[0] - 1) + dila_kernel.shape[0]*margin]}}
        fig = go.Figure(data=[go.Bar(y=dila_kernel, width=dila_kernel.shape[0] * bar_scale)], layout=layout)
    else:
        layout = {'title': {'text': f'Kernel: {kernel}'},
                  'xaxis': {'range': 
                            [-kernel.shape[0] * margin,
                             (kernel.shape[0] - 1) + kernel.shape[0] * margin]}}
        fig = go.Figure(data=[go.Bar(y=kernel, width=kernel.shape[0] * bar_scale)], layout=layout)
    return fig


@app.callback(
    [Output(component_id='ts_trans_plot', component_property='figure'),
     Output(component_id='ppv-value', component_property='children'),
     Output(component_id='min-value', component_property='children'),
     Output(component_id='max-value', component_property='children'),],
    [Input(component_id='current-TS', component_property='children'),
     Input(component_id='current-kernel', component_property='children'),
     Input(component_id='kernel_bias', component_property='value'),
     Input(component_id='kernel_padding', component_property='value'),
     Input(component_id='kernel_stride', component_property='value'),
     ]
)
def plot_trans_ts(json_data, kernel, bias, padding, stride):

    dff = pd.read_json(json_data, orient='values').to_numpy().flatten()
    kernel = np.array(json.loads(kernel)['dila_kernel'])
    bias = float(bias)
    padding = int(padding)
    stride = int(stride)
    transformed = utls.apply_kernel(dff, kernel, bias, padding, stride)
    ts_length = dff.shape[0]
    layout = {'title': {'text': 'Transformed time series'},
              'xaxis': {'range': [0, ts_length]}}
    ppv = utls.ppv(transformed)
    max_value = np.max(transformed)
    min_value = np.min(transformed)
    return go.Figure(data=[go.Scatter(y=transformed)], layout=layout), f"ppv = {ppv:.2f}", f"min = {min_value:.2f}", f"max = {max_value:.2f}"


@app.callback(
    [Output('all-data', 'children'),
     Output('ts-select', 'max'),
     Output('ts-select', 'value'),],
    [Input('upload-data', 'contents')],)
def upload_data(list_of_contents):
    if list_of_contents is not None:
        print("Load uploaded dataset")
        content_type, content_string = list_of_contents.split(',')
        decoded = base64.b64decode(content_string)
        c = io.BytesIO(decoded)
        data = pd.DataFrame(np.loadtxt(c, delimiter=',')[:, 1:].T)
    else:
        print("Load default dataset")
        datafile='data/CMJ/JumpResampled/JumpResampled_TRAIN'
        data = pd.DataFrame(np.loadtxt(datafile, delimiter=',')[:, 1:].T)
    print(data.shape)
    return data.to_json(orient='values'), data.shape[1]-1, 0



app.layout = html.Div(children=[
    html.H4(children='TS kernel visualization'),
    # Div to store the current TS
    html.Div(id='current-TS', style={'display': 'none'}),
    html.Div(id='current-kernel', style={'display': 'none'}),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files'),
                html.P('Note, the expected format is a csv file, each time series on a seperate line and with the first entry beeing the class label (will be omitted).',)
            ]),
            style={
                'width': '100%',
                'height': '120px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
        multiple=False
        ),
        html.Div(id='all-data', style={'display': 'none'}),
    ]),
    # Time series plot
    html.Div([
        html.Div([
            html.Div(children=[
                daq.BooleanSwitch(
                    id="normalize_ts_switch",
                    on=False,
                    label='Normalize time series',
                ),
                daq.BooleanSwitch(
                    id="test_ts_switch",
                    on=False,
                    label='Use test time series',
                ),
            ]),

            # Test time series

            html.Div([  # div to keep spacing correct
                html.Div(
                    id="test-ts-container",
                    children=[
                        dcc.Input(
                            id="test-ts",
                            placeholder='Enter your test time series',
                            type='text',
                            value='1,1,1,1,1,1,1,1,1,1,1,1,1',
                            debounce=True
                        )],),

                html.Div(
                    id="ts-select-container",
                    children=[
                        html.P(["Select time series"],
                               style={
                                   'width': '100%',
                                   'textAlign': 'center'}),
                        dcc.Slider(
                            id='ts-select',
                            min=-0,
                            step=1,
                            value=0),
                        html.Div(id='ts-slider-output-container',
                                 style={
                                     'width': '100%',
                                     'textAlign': 'center'}
                                 ),
                    ]),
            ],
                     style={'margin-top': '1em'},)
        ], className="two columns"),

        html.Div([
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
                ),
                daq.BooleanSwitch(
                    id="kernel_centering_chkb",
                    on=False,
                    label='Center Kernel',
                    style={'alignItems': 'flex-star'}
                ),
                daq.BooleanSwitch(
                    id="kernel-plot-with-dila",
                    on=False,
                    label='Plot with dilation',
                    style={'alignItems': 'flex-star'}
                ),
            ]),
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
        html.Div([
            html.H6("Pooling values"),
            html.Div(id="ppv-value"),
            html.Div(id="max-value"),
            html.Div(id="min-value"),
        ], className="two columns"),
        html.Div([
            dcc.Graph(
                id="ts_trans_plot")
        ], className="ten columns"),
    ], className="row"),
])

if __name__ == '__main__':
    app.run_server(debug=True)
