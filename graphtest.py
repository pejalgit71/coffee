import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample Data
data = {
    'Coffee Type': ['Espresso', 'Latte', 'Cappuccino', 'Americano', 'Mocha'],
    'Sales': [250, 400, 300, 200, 100],
}

# Create DataFrame
df = pd.DataFrame(data)

# Find the best and worst sales
best_sale = df[df['Sales'] == df['Sales'].max()]
worst_sale = df[df['Sales'] == df['Sales'].min()]

# Title and description
st.title("XYZ Coffee Shop Sales Analysis")
st.write("This analysis displays the best and worst coffee sales for XYZ Coffee Shop, with a bar chart and pie chart comparison.")

# Plot the Bar Chart
st.subheader("Bar Chart of Coffee Sales")
fig, ax = plt.subplots()
ax.bar(df['Coffee Type'], df['Sales'], color='skyblue')
ax.set_xlabel('Coffee Type')
ax.set_ylabel('Sales')
ax.set_title('Coffee Sales Comparison')

# Highlight best and worst sales
ax.bar(best_sale['Coffee Type'], best_sale['Sales'], color='green', label='Best Sale')
ax.bar(worst_sale['Coffee Type'], worst_sale['Sales'], color='red', label='Worst Sale')

# Add legend
ax.legend()

# Show bar chart in Streamlit
st.pyplot(fig)

# Plot the Pie Chart
st.subheader("Pie Chart of Coffee Sales")
fig2, ax2 = plt.subplots()
ax2.pie(df['Sales'], labels=df['Coffee Type'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#ff6666'])
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show pie chart in Streamlit
st.pyplot(fig2)

# Display Best and Worst Sale Information
st.write("### Best Sale")
st.write(best_sale)

st.write("### Worst Sale")
st.write(worst_sale)
