from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Enter your data to see the results"),
    html.Div([
        html.H2('Enter your weight in kg'),
        dcc.Input(id = 'weight', value = 95, type = 'number'),
        html.H2('Enter your height in cm'),
        dcc.Input(id = 'height', value = 200, type = 'number'),
        html.H2('Enter your age in years'),
        dcc.Input(id = 'age', value = 50, type = 'number'),
        html.H2('Enter your gender'),
        dcc.RadioItems(options = [{'label': 'Male', 'value': 'm'},{'label': 'Female', 'value': 'f'}],
                       value = 'm',
                       id = 'gender')
    ]),
    html.Br(),
    html.H1("Your estimated basal metabolic rate is: "),
    html.H2(id = 'bmr'),

])


@app.callback(
    Output(component_id = 'bmr'   , component_property = 'children'),
    Input(component_id  = 'weight', component_property = 'value'),
    Input(component_id  = 'height', component_property = 'value'),
    Input(component_id  = 'age'   , component_property = 'value'),
    Input(component_id  = 'gender'   , component_property = 'value')
)
def update_output_div(weight, height, age, gender):
    if gender == 'm':
        rval = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    if gender == 'f':
        rval = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    return rval

if __name__ == '__main__':
    app.run_server(debug=True, host = '127.0.0.1')

