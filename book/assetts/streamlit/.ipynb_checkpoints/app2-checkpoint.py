import streamlit as st
import pandas as pd
import numpy as np

cb = st.checkbox("Did you check the box")
if cb:
    st.write("Yes you did! :-)")
else:
    st.write("No you didn't :-(")


rb = st.radio(
     "Pick an option",
     ('Option 1', 'Option 2', 'Option 3'))
st.write("you picked " + rb)


## Adding a chart

nsims = st.number_input('Put in a number of sims', value = 10)
if nsims < 10 : 
    st.write("Pick a bigger number")
else :
    chart_data = pd.DataFrame(
         np.random.randn(np.round(nsims), 2),
         columns=['a', 'b'])
    st.line_chart(chart_data)
