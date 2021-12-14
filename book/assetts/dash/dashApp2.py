import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])


@app.callback([Output(component_id='my-output', component_property='children')],
    [Input(component_id='my-input', component_property='value')])

def update_output_div(input_value):
    return 'Output: {}'.format(input_value)


app.run_server(debug=True)