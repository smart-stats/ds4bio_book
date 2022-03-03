import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np

## Define the app
app = dash.Dash(__name__)

## Read in our data
dat = pd.read_csv("kirby21.csv").drop(['Unnamed: 0'], axis = 1)
dat = dat.assign(id_char = "id_" + dat.id.astype(str))
fig = px.bar(dat, x = "id_char", y = "volume", color = "roi")

app.layout = html.Div(children=[
    html.H1(children='Subject level compositional data'),

    dcc.Graph(
        id = 'graph',
        figure = fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, host = '127.0.0.1')

