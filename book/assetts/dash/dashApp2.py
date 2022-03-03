from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd


dat = pd.read_csv('kirby21AllLevels.csv')
dat = dat.loc[dat['type'] == 1].groupby(["roi", "level"])['volume'].mean().reset_index()

app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(options = [
            {'label' : '1', 'value' : 1},
            {'label' : '2', 'value' : 2},
            {'label' : '3', 'value' : 3},
            {'label' : '4', 'value' : 4},
            {'label' : '5', 'value' : 5}
        ],
        value = 1, id = 'input-level'
                ),
    dcc.Graph(id = 'output-graph')
])


@app.callback(
    Output('output-graph', 'figure'),
    Input('input-level', 'value'))
def update_figure(selected_level):
    subdat = dat.loc[dat['level'] == int(selected_level)].sort_values(by = ['volume'])
    fig = px.bar(subdat, x='roi', y='volume')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host = '127.0.0.1')



