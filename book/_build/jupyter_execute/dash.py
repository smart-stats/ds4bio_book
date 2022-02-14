#!/usr/bin/env python
# coding: utf-8

# # Dash
# 
# Dash is a framework for developing dashboards and creating web based data apps in R, python or Julia. Dash is more popular among the python focused, while shiny, a related platform, is more popular among R focused, but also applies much more broadly than just for R apps.
# 
# Because we're focusing on python as our base language, we'll focus on dash. There's a wonderful set of tutorials [here](https://dash.plotly.com/introduction). Follow the instructions there on installation. We'll build a simple app here building on that tutorial. Let's take the first plotly example and use one of our examples.
# 
# 
# 

# Put this code below in a file, say dashApp.py (in the github repo it's in `assetts/dash`), then run it with `python dashApp.py`. If all has gone well, your app should be running locally at `http://127.0.0.1:8050/` (so visit that site in a browser). 
# 
# 
# ```
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.express as px
# import pandas as pd
# import numpy as np
# 
# ## Define the app
# app = dash.Dash(__name__)
# 
# ## Read in our data
# dat = pd.read_csv("PATH TO THE DATA/kirby21.csv").drop(['Unnamed: 0'], axis = 1)
# dat = dat.assign(id_char = "id_"+dat.id.astype(str))
# 
# ## Produce the figure
# fig = px.bar(dat, x = "id_char", y = "volume", color = "roi")
# 
# ## This creates the layout of the page
# app.layout = html.Div(children=[
#     ## HTML elements added with html.method
#     html.H1(children='Subject level compositional data'),
#     
#     ## Dynamic graph is added with dcc.METHOD (dcc = dynamic core component)
#     dcc.Graph(
#         id = 'graph',
#         figure = fig
#     )
# ])
# 
# ## This runs the server
# if __name__ == '__main__':
#     app.run_server(debug=True)
# ```
# 
# The resulting website, running at `http://127.0.0.1:8050/` looks like this for me:
# 
# ![Grahpic](assetts/dashExample.png)
# 
# Again, `127.0.0.1` is the localhost address and `:8050` is the port. You can change the port in the `.run_server` method. But, we want fancier apps that call a server and return calculations back to us (so-called callbacks). 
# 
# 
