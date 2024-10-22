import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet

# Title of the App
st.title("XstreamCoffee.net Sales Prediction")

# Upload CSV file
st.subheader("Upload your Coffee Sales Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV
    data = pd.read_csv(uploaded_file)
    
    # Show preview of the uploaded file
    st.subheader("Sales Data Preview")
    st.write(data.head())

    # Handling KeyError: 'Date'
    # Check if 'Date' column exists, if not, prompt the user to select the correct column
    if 'Date' not in data.columns:
        date_column = st.selectbox("Select the Date column from the dataset", data.columns)
    else:
        date_column = 'Date'

    # Convert the selected date column to datetime
    try:
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
    except Exception as e:
        st.error(f"Error converting {date_column} to datetime: {e}")
        st.stop()

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    
    # Feature Selection
    st.write("Selected Features: 'Date', 'Coffee Flavor', 'Sales Quantity'")
    
    # Group by 'Coffee Flavor' and 'Date' to aggregate sales
    sales_data = data.groupby([data.index, 'Coffee Flavor']).sum().reset_index()
    st.write(sales_data.head())
    
    # User Input for Coffee Flavor and Date Range
    coffee_flavors = sales_data['Coffee Flavor'].unique()
    selected_flavor = st.selectbox("Select Coffee Flavor to Visualize", coffee_flavors)

    # Date range picker for filtering
    start_date = st.date_input("Start Date", value=data.index.min().date())
    end_date = st.date_input("End Date", value=data.index.max().date())

    # Filter data for selected flavor and date range
    flavor_data = sales_data[(sales_data['Coffee Flavor'] == selected_flavor) & 
                             (sales_data['Date'] >= pd.to_datetime(start_date)) & 
                             (sales_data['Date'] <= pd.to_datetime(end_date))]

    # Extract month from Date for Trellis Display and other visualizations
    flavor_data['month'] = flavor_data['Date'].dt.month

    # Display a plot of the sales data
    st.subheader("Sales Data Visualization")
    chart_type = st.selectbox("Select Chart Type", ['Line Chart', 'Bar Chart', 'Scatter Plot', '3D Scatter (Grand Tour)', 'Parallel Coordinates', 'Trellis Display', 'Mosaic Display'])

    # Plot the sales trend for the selected flavor
    plt.figure(figsize=(10, 6))
    
    # Line Chart
    if chart_type == 'Line Chart':
        plt.plot(flavor_data['Date'], flavor_data['Sales Quantity'], label=selected_flavor)

    # Bar Chart
    elif chart_type == 'Bar Chart':
        plt.bar(flavor_data['Date'], flavor_data['Sales Quantity'], label=selected_flavor)

    # Scatter Plot
    elif chart_type == 'Scatter Plot':
        fig = px.scatter(flavor_data, x='Date', y='Sales Quantity', title=f"Scatter Plot of {selected_flavor}")
        st.plotly_chart(fig)

    # 3D Scatter Plot (Grand Tour for high-dimensional data)
    elif chart_type == '3D Scatter (Grand Tour)':
        flavor_data['day'] = flavor_data['Date'].dt.day
        fig = px.scatter_3d(flavor_data, x='day', y='month', z='Sales Quantity', color='Sales Quantity', title="3D Sales Grand Tour")
        st.plotly_chart(fig)

    # Parallel Coordinates Plot
    elif chart_type == 'Parallel Coordinates':
        flavor_data['day'] = flavor_data['Date'].dt.day
        flavor_data['year'] = flavor_data['Date'].dt.year
        fig = px.parallel_coordinates(flavor_data, color='Sales Quantity', dimensions=['day', 'month', 'year'], title=f"Parallel Coordinates for {selected_flavor}")
        st.plotly_chart(fig)

    # Trellis Display (Fixed - Added 'month' column)
    elif chart_type == 'Trellis Display':
        fig = px.scatter(flavor_data, x='Date', y='Sales Quantity', facet_col='month', title=f"Trellis Display for {selected_flavor}")
        st.plotly_chart(fig)

    # Mosaic Display (Trellis-like)
    elif chart_type == 'Mosaic Display':
        fig = px.treemap(flavor_data, path=['month', 'Sales Quantity'], values='Sales Quantity', title=f"Mosaic Display for {selected_flavor}")
        st.plotly_chart(fig)

    # Matplotlib-based visualization
    if chart_type in ['Line Chart', 'Bar Chart']:
        plt.title(f"Sales Trend for {selected_flavor}")
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    # Feature Engineering and Model Training
    st.subheader("Feature Engineering and Model Training")

    # Add additional features (Promotions, Prices, Holidays)
    promotions = st.number_input("Enter Promotions Impact (percentage)", min_value=0, max_value=100, value=0)
    prices = st.number_input("Enter Price of Coffee", min_value=0.0, value=0.0, format="%.2f")
    holidays = st.number_input("Enter Holidays Impact (percentage)", min_value=0, max_value=100, value=0)

    # Creating a new feature for modeling
    sales_data['day'] = sales_data['Date'].dt.day
    sales_data['month'] = sales_data['Date'].dt.month
    sales_data['year'] = sales_data['Date'].dt.year

    # Adjust sales based on promotions and holidays
    sales_data['Adjusted Sales'] = sales_data['Sales Quantity'] * (1 - promotions / 100) * (1 - holidays / 100)

    # Split the data into training and testing
    X = sales_data[['day', 'month', 'year']]
    y = sales_data['Adjusted Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Show Model Evaluation
    st.subheader("Model Performance")
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error: {mae:.2f}")

    # Forecast Future Sales using Facebook Prophet
    st.subheader("Future Sales Prediction with Facebook Prophet")
    
    # Prepare data for Prophet
    prophet_data = flavor_data[['Date', 'Sales Quantity']].rename(columns={'Date': 'ds', 'Sales Quantity': 'y'})
    
    # Add promotions and holidays as regressors
    prophet_data['promotions'] = promotions
    prophet_data['holiday_events'] = holidays

    # Initialize and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.add_regressor('promotions')
    prophet_model.add_regressor('holiday_events')
    prophet_model.fit(prophet_data)

    # User input for prediction duration
    prediction_days = st.slider("Select Prediction Period (days)", min_value=1, max_value=150, value=30)
    
    # Create future dataframe for prediction
    future_dates = prophet_model.make_future_dataframe(periods=prediction_days)
    future_dates['promotions'] = promotions
    future_dates['holiday_events'] = holidays

    # Make predictions
    forecast = prophet_model.predict(future_dates)

    # Show predictions
    st.subheader("Predicted Future Sales")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days))

    # Plot future sales with chart type selection
    st.subheader("Future Sales Visualization")
    future_chart_type = st.selectbox("Select Future Prediction Chart Type", ['Line Chart', 'Bar Chart', 'Scatter Plot', '3D Scatter (Grand Tour)', 'Parallel Coordinates', 'Trellis Display', 'Mosaic Display'])

    plt.figure(figsize=(10, 6))

    # Future predictions chart type visualization
    if future_chart_type == 'Line Chart':
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Sales', color='green')
    elif future_chart_type == 'Bar Chart':
        plt.bar(forecast['ds'], forecast['yhat'], label='Predicted Sales', color='green')
    elif future_chart_type == 'Scatter Plot':
        fig = px.scatter(forecast, x='ds', y='yhat', title="Future Sales Scatter Plot")
        st.plotly_chart(fig)
    elif future_chart_type == '3D Scatter (Grand Tour)':
        forecast['day'] = forecast['ds'].dt.day
        forecast['month'] = forecast['ds'].dt.month
        fig = px.scatter_3d(forecast, x='day', y='month', z='yhat', color='yhat', title="3D Future Sales Grand Tour")
        st.plotly_chart(fig)
    elif future_chart_type == 'Parallel Coordinates':
        forecast['day'] = forecast['ds'].dt.day
        forecast['month'] = forecast['ds'].dt.month
        forecast['year'] = forecast['ds'].dt.year
        fig = px.parallel_coordinates(forecast, color='yhat', dimensions=['day', 'month', 'year'], title="Parallel Coordinates for Future Predictions")
        st.plotly_chart(fig)
    elif future_chart_type == 'Trellis Display':
        forecast['ds'] = pd.to_datetime(forecast['ds'])  # Ensure 'ds' is datetime
        forecast['month'] = forecast['ds'].dt.month      # Create 'month' column
        
        fig = px.scatter(forecast, x='ds', y='yhat', facet_col='month', title="Trellis Display for Future Predictions")
        st.plotly_chart(fig)
    elif future_chart_type == 'Mosaic Display':
        fig = px.treemap(forecast, path=['month', 'yhat'], values='yhat', title="Mosaic Display for Future Predictions")
        st.plotly_chart(fig)

    # Displaying the plot
    if future_chart_type in ['Line Chart', 'Bar Chart']:
        plt.title("Predicted Future Sales")
        plt.xlabel('Date')
        plt.ylabel('Predicted Sales Quantity')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    # Show forecasted values
    st.subheader("Forecast Visualization")
    fig1 = px.line(forecast, x='ds', y='yhat', title='Forecasted Sales')
    st.plotly_chart(fig1)

    # Adding option to download forecasted data
    st.subheader("Download Forecast Data")
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecasted Sales Data",
        data=csv,
        file_name='forecasted_sales_data.csv',
        mime='text/csv',
    )
