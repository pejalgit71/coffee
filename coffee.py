import streamlit as st
st.set_page_config(layout="wide")
import warnings
import pandas as pd
from datetime import datetime
import warnings
import os, fnmatch
from PIL import Image
import glob
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


formside = st.sidebar.form("side_form")
choose = formside.radio("Choose Report",["Report", "Graph"], index=None)
formside.form_submit_button("Submit")

coffee_df = pd.read_csv("coffee.csv")
coffee_df = pd.DataFrame(coffee_df)

coffee_df = coffee_df.reset_index(drop=True)
coffee_df.index = coffee_df.index+1

st.title("My Coffee Shop SI")



if (choose == "Report"):
    # Multi-select for columns
    columns = st.multiselect("Select columns to display", coffee_df.columns.tolist(), default=coffee_df.columns.tolist())

    # Display table with selected columns
    st.dataframe(coffee_df[columns])
    
    selected_product = st.selectbox("Select a product to filter by", coffee_df['Product'].unique())

    # Filter DataFrame based on selected product
    filtered_df = coffee_df[coffee_df['Product'] == selected_product]
    st.write("Filtered Data")
    st.dataframe(filtered_df[columns])

if (choose == "Graph"):

     # Multi-select for columns
    columns = st.multiselect("Select columns to display", coffee_df.columns.tolist(), default=coffee_df.columns.tolist())

    # Display table with selected columns
    st.dataframe(coffee_df[columns])
    
    selected_product = st.selectbox("Select a product to filter by", coffee_df['Product'].unique())

    # Filter DataFrame based on selected product
    filtered_df = coffee_df[coffee_df['Product'] == selected_product]
    st.write("Filtered Data")
    st.dataframe(filtered_df[columns])

    # Display total sales based on selected columns
    if 'Total Revenue (USD)' in columns:
        total_sales = coffee_df['Total Revenue (USD)'].sum()
        st.write(f"Total Sales for the displayed data: ${total_sales:.2f}")
    
    # --- New code to display bar chart for total sales ---
    st.write("Bar Chart: Total Sales by Product")
    
    # Group data by Product and sum the total revenue
    sales_by_product = coffee_df.groupby('Product')['Total Revenue (USD)'].sum()
    
    # Plotting the bar chart using Matplotlib
    fig, ax = plt.subplots()
    sales_by_product.plot(kind='bar', ax=ax)
    ax.set_ylabel("Total Sales (USD)")
    ax.set_title("Total Sales by Product")
    
    # Display the bar chart in the Streamlit app
    st.pyplot(fig)
