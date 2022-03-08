from dash import Dash, dcc

app = Dash(__name__)


app.layout = dcc.Markdown('''
# Section 1
## Section 2
### Section 3

1. Numbered lists
2. Second item

* Bulleted list
* Second item
''')


if __name__ == '__main__':
    app.run_server(host = '127.0.0.1')