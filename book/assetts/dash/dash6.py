import dash_leaflet as dl
import dash_leaflet.express as dlx
import dash
from dash import Dash, html, dcc, Output, Input
from dash_extensions.javascript import assign
from dash.dependencies import Input, Output, State
import os

app = Dash()  
app.layout = html.Div([
    dl.Map(id = "output-state"),
    dcc.Input(id = 'lat', value = 39.29817905327218, type = 'number'),
    dcc.Input(id = 'long', value = -76.5902156598373, type = 'number'),
    dcc.Input(id = 'zoom', value = 11, type = 'number'),
    html.Button('Submit', id='submit-button')
])

@app.callback(Output('output-state', 'children'),
              Input('submit-button', 'n_clicks'),
              State('lat', 'value'),
              State('long', 'value'),
              State('zoom', 'value'))
def update_output(n_clicks, lat, long, zoom):
    if n_clicks is not None:
        return dl.Map([dl.TileLayer()], 
                        center = (lat, long), 
                        zoom = zoom,
                        style={'width': '100%', 'height': '75vh', 'margin': "auto", "display": "block"})

