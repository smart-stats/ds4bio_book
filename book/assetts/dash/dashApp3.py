from datetime import date
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dropdown"),
    dcc.Dropdown(
       options=[
           {'label': 'Type 1', 'value': 1},
           {'label': 'Type 2', 'value': 2},
           {'label': 'Type 3', 'value': 3},
       ],
       value = 2
    ),
    html.H1("Checklist"),
    dcc.Checklist(       
        options=[
           {'label': 'Type 1', 'value': 1},
           {'label': 'Type 2', 'value': 2},
           {'label': 'Type 3', 'value': 3},
       ]
    ), 
    html.H1("Slider"),
    dcc.Slider(min = 0, max = 20, step = 5, value = 10, id='slider'),
    html.H1("Date picker"),
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed = date(1995, 8, 5),
        max_date_allowed = date(2017, 9, 19),
        initial_visible_month=date(2017, 8, 5),
        date=date(2017, 8, 25)
    )

])



if __name__ == '__main__':
    app.run_server(host = '127.0.0.1')