import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import time

# Title of the Streamlit app
st.title('Weather Patterns and Solar Energy')
st.markdown("""
**Welcome to the Weather-Solar Analysis Tool**

This app delves into the relationship between local weather patterns and solar energy production. It provides insights into how these patterns can help predict and optimize solar energy output based on weather forecasts.

**Key Focus:**
- **Shortwave Radiation Sum**: The primary variable for analyzing solar radiation in both daily and hourly datasets.
- **Variable Interactions**: The app explores interactions between weather conditions—such as cloud cover, humidity, and precipitation—and their impact on solar energy. Understanding these dynamics helps enhance energy management and planning strategies.

**Suggested Variables to Explore:**
- **Temperature**: Assess how temperature fluctuations impact solar efficiency.
- **Daylight Duration**: Explore the link between daylight hours and solar output.
- **Sunshine Duration**: Analyze sunlight exposure's effect on energy generation.
- **Precipitation and Humidity**: Study moisture and rain impacts on panel efficiency.
- **Weather Codes**: Investigate how various weather conditions affect solar production.
""")            
           

with st.sidebar:
    with st.expander("Data Overview", expanded=False):
        st.markdown("""
        The dataset is divided into four data frames, originating from two different locations:
        - **Provo, Utah**
        - **Redmond, Washington**

        Each location has data available in both daily and hourly intervals, covering the period from 2008 to 2022.

        ### Daily Data Frames Variables:
        - `date`: The date of data recording.
        - `weather_code`: Code indicating the type of weather observed.
        - `temperature_2m_mean`: Average temperature at 2 meters above ground.
        - `daylight_duration`: Total duration of daylight.
        - `sunshine_duration`: Total duration of sunshine.
        - `precipitation_sum`: Total precipitation.
        - `rain_sum`: Total rainfall.
        - `snowfall_sum`: Total snowfall.
        - `shortwave_radiation_sum`: Sum of shortwave radiation.

        ### Hourly Data Frames Variables:
        - `date`: The date of data recording.
        - `temperature_2m`: Temperature at 2 meters above ground.
        - `relative_humidity_2m`: Relative humidity at 2 meters above ground.
        - `weather_code`: Code indicating the type of weather observed.
        - `cloud_cover`: Percentage of the sky occluded by clouds.
        - `shortwave_radiation`: Shortwave radiation received.
        - `direct_normal_irradiance`: Direct normal solar irradiance.
        - `global_tilted_irradiance`: Global solar irradiance on a tilted surface.
        """)

with st.sidebar:
    st.header("Machine Learning Analysis")
    st.markdown("""
    **Model Insights & Performance**
    Our app features two machine learning widgets that utilize Random Forest Regression to analyze data. These widgets calculate the mean square error (MSE) of the predictions to evaluate model accuracy. Additionally, they highlight the most influential features impacting the predictions. The training and execution time for each model is also displayed, providing insights into the efficiency of the algorithm.
    """)

# Load DataFrames and immediately parse dates as UTC
df_daily_provo = pd.read_csv('daily_data_provo.csv', parse_dates=['date'])
df_hourly_provo = pd.read_csv('hourly_data_provo.csv', parse_dates=['date'])
df_redmond_day = pd.read_csv('daily_data_redmond.csv', parse_dates=['date'])
df_redmond_hour = pd.read_csv('hourly_data_redmond.csv', parse_dates=['date'])

# Ensure datetime objects are in UTC
def ensure_utc(df, date_column):
    if df[date_column].dt.tz is None:
        df[date_column] = df[date_column].dt.tz_localize('UTC', ambiguous='infer')
    else:
        df[date_column] = df[date_column].dt.tz_convert('UTC')
    return df

df_daily_provo = ensure_utc(df_daily_provo, 'date')
df_hourly_provo = ensure_utc(df_hourly_provo, 'date')
df_redmond_day = ensure_utc(df_redmond_day, 'date')
df_redmond_hour = ensure_utc(df_redmond_hour, 'date')

# Streamlit widget to select a column to compare
column_to_compare = st.selectbox(
    'Select a column to compare with Shortwave Radiation Sum (Provo Daily Data):',
    [col for col in df_daily_provo.columns if col != 'shortwave_radiation_sum']
)

# Function to create and show the plot
def plot_data_provo(df, x_col, y_col='shortwave_radiation_sum'):
    if x_col == 'weather_code':
        mean_values = df.groupby(x_col)[y_col].mean().reset_index()
        # Use line plot for weather_code comparison
        fig = px.line(mean_values, x=x_col, y=y_col, color_discrete_sequence=['blue'],
                      title=f'Relationship between {x_col} and {y_col}',
                      labels={x_col: 'Weather Condition', y_col: y_col},
                      template="simple_white")
    else:
       fig = px.scatter(df, x=x_col, y=y_col, title=f'Relationship between {x_col} and {y_col}, Provo Daily Data',
                     labels={x_col: x_col, y_col: y_col}, template="simple_white")
    st.plotly_chart(fig, use_container_width=True)
    #print(df_daily_provo.head())

# Call the plotting function with user-selected column
plot_data_provo(df_daily_provo, column_to_compare)

column_to_compare_hp = st.selectbox(
    'Select a column to compare with Shortwave Radiation Sum (Provo Hourly Data):',
    [col for col in df_hourly_provo.columns if col != 'shortwave_radiation']
)

# Function to create and show the plot
def plot_data_provo_hour(df, x_col, y_col='shortwave_radiation'):
    if x_col == 'weather_code':
        mean_values_h = df.groupby(x_col)[y_col].mean().reset_index()
        # Use line plot for weather_code comparison
        fig = px.line(mean_values_h, x=x_col, y=y_col, color_discrete_sequence=['blue'],
                      title=f'Relationship between {x_col} and {y_col}',
                      labels={x_col: 'Weather Condition', y_col: y_col},
                      template="simple_white")
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=f'Relationship between {x_col} and {y_col}, Provo Hourly Data',
                     labels={x_col: x_col, y_col: y_col}, template="simple_white") 
        st.plotly_chart(fig, use_container_width=True)
    #print(df_daily_provo.head())

# Call the plotting function with user-selected column
plot_data_provo_hour(df_hourly_provo, column_to_compare_hp)

column_to_compare_r = st.selectbox(
    'Select a column to compare with Shortwave Radiation Sum (Redmond Daily Data):',
    [col for col in df_redmond_day.columns if col != 'shortwave_radiation_sum']
)

# Function to create and show the plot
def plot_data_redmond(df, x_col, y_col='shortwave_radiation_sum'):
    if x_col == 'weather_code':
        mean_values = df.groupby(x_col)[y_col].mean().reset_index()
        # Use line plot for weather_code comparison
        fig = px.line(mean_values, x=x_col, y=y_col, color_discrete_sequence=['blue'],
                      title=f'Relationship between {x_col} and {y_col}',
                      labels={x_col: 'Weather Condition', y_col: y_col},
                      template="simple_white")
    else:
       fig = px.scatter(df, x=x_col, y=y_col, title=f'Relationship between {x_col} and {y_col}, Redmond Daily Data',
                     labels={x_col: x_col, y_col: y_col}, template="simple_white")
    st.plotly_chart(fig, use_container_width=True)
    #print(df_daily_provo.head())

# Call the plotting function with user-selected column
plot_data_redmond(df_redmond_day, column_to_compare_r)

column_to_compare_hr = st.selectbox(
    'Select a column to compare with Shortwave Radiation Sum (Redmond Hourly Data):',
    [col for col in df_redmond_hour.columns if col != 'shortwave_radiation']
)

# Function to create and show the plot
def plot_data_redmond_hour(df, x_col, y_col='shortwave_radiation'):
    fig = px.scatter(df, x=x_col, y=y_col, title=f'Relationship between {x_col} and {y_col}, Redmond Hourly Data',
                     labels={x_col: x_col, y_col: y_col}, template="simple_white") 
    st.plotly_chart(fig, use_container_width=True)
    #print(df_daily_provo.head())

# Call the plotting function with user-selected column
plot_data_redmond_hour(df_redmond_hour, column_to_compare_hr)


st.sidebar.title("Daily Data: Random Forest Controls")
data_options = ['Provo Daily Data', 'Redmond Daily Data']
selected_data = st.sidebar.selectbox('Choose the dataset:', data_options)

# Map selection to DataFrame
df_map = {'Provo Daily Data': df_daily_provo, 'Redmond Daily Data': df_redmond_day}
df_selected = df_map[selected_data]

max_depth_day = st.sidebar.slider('Select max depth of the day tree:', 1, 20, 10, step=1)
train_model_day = st.sidebar.button('Train Day Model')

if train_model_day:
    start_time = time.time()
    
    X = df_selected[['temperature_2m_mean', 'daylight_duration', 'sunshine_duration', 'precipitation_sum', 'rain_sum', 'snowfall_sum', 'weather_code']]
    y = df_selected['shortwave_radiation_sum']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train Random Forest Regressor
    regressor_day = RandomForestRegressor(max_depth=max_depth_day, random_state=42)
    regressor_day.fit(X_train, y_train)

    # Predicting and checking the performance
    y_pred = regressor_day.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.sidebar.write(f'Day Model MSE: {mse}')

    # Displaying feature importances
    feature_importance_day = pd.DataFrame({'Feature': X.columns, 'Importance': regressor_day.feature_importances_})
    fig_day = px.bar(feature_importance_day, x='Importance', y='Feature', title='Day Model Feature Importance', orientation='h')
    st.plotly_chart(fig_day, use_container_width=True)

    # Calculate and display elapsed time
    elapsed_time_day = time.time() - start_time
    st.sidebar.write(f'Day Model Training and Testing Time: {elapsed_time_day:.2f} seconds')

st.sidebar.title("Hourly Data: Random Forest Controls")
data_options_hourly = ['Provo Hourly Data', 'Redmond Hourly Data']
selected_data_hourly = st.sidebar.selectbox('Choose the hourly dataset:', data_options_hourly)

# Map selection to DataFrame
df_map_hourly = {'Provo Hourly Data': df_hourly_provo, 'Redmond Hourly Data': df_redmond_hour}
df_selected_hourly = df_map_hourly[selected_data_hourly]

max_depth_hour = st.sidebar.slider('Select max depth of the hour tree:', 1, 20, 10, step=1)
train_model_hourly = st.sidebar.button('Train Hourly Model')

if train_model_hourly:
    start_time_hourly = time.time()

    X_hourly = df_selected_hourly[['temperature_2m', 'relative_humidity_2m', 'weather_code', 'cloud_cover', 'direct_normal_irradiance']]
    y_hourly = df_selected_hourly['shortwave_radiation']

    # Split data into training and testing sets
    X_train_hourly, X_test_hourly, y_train_hourly, y_test_hourly = train_test_split(X_hourly, y_hourly, test_size=0.2, random_state=42)

    # Create and train Random Forest Regressor
    regressor_hourly = RandomForestRegressor(max_depth=max_depth_hour, random_state=42)
    regressor_hourly.fit(X_train_hourly, y_train_hourly)

    # Predicting and checking the performance
    y_pred_hourly = regressor_hourly.predict(X_test_hourly)
    mse_hourly = mean_squared_error(y_test_hourly, y_pred_hourly)
    st.sidebar.write(f'Hourly Model MSE: {mse_hourly}')

    # Displaying feature importances
    feature_importance_hourly = pd.DataFrame({'Feature': X_hourly.columns, 'Importance': regressor_hourly.feature_importances_})
    fig_hourly = px.bar(feature_importance_hourly, x='Importance', y='Feature', title='Hourly Model Feature Importance', orientation='h')
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Calculate and display elapsed time
    elapsed_time_hourly = time.time() - start_time_hourly
    st.sidebar.write(f'Hourly Model Training and Testing Time: {elapsed_time_hourly:.2f} seconds')

# Sidebar for DataFrame selection
st.sidebar.title("DataFrame Selection for Daily Data")
st.sidebar.markdown("""
- Warning: When you select your start date, the app will give an error, once you select your end date it will be resolved. 
""")    
dataframe_options = {
    "Provo Daily Data": df_daily_provo,
    "Redmond Daily Data": df_redmond_day
}
selected_dataframe_name = st.sidebar.selectbox("Select the data set:", list(dataframe_options.keys()))
df_selected = dataframe_options[selected_dataframe_name]

# Updating existing filters to use selected DataFrame
st.sidebar.title("Filter Options for Daily Data")

# Date range filter
if not df_selected.empty:
    start_date, end_date = st.sidebar.date_input(
        "Select date range",
        [df_selected['date'].min(), df_selected['date'].max()],
        min_value=df_selected['date'].min(),
        max_value=df_selected['date'].max()
    )
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    df_filtered = df_selected[(df_selected['date'] >= start_date) & 
                                (df_selected['date'] <= end_date)]

    # Weather code filter
    weather_options = df_selected['weather_code'].dropna().unique()
    selected_weather = st.sidebar.multiselect('Select weather conditions', options=weather_options)
    if selected_weather:
        df_filtered = df_filtered[df_filtered['weather_code'].isin(selected_weather)]

    # Temperature range selector
    if not df_filtered.empty:
        min_temp, max_temp = st.sidebar.slider("Select temperature range (°F)", 
                                               float(df_filtered['temperature_2m_mean'].min()), 
                                               float(df_filtered['temperature_2m_mean'].max()), 
                                               (float(df_filtered['temperature_2m_mean'].min()), 
                                               float(df_filtered['temperature_2m_mean'].max())))
        df_filtered = df_filtered[(df_filtered['temperature_2m_mean'] >= min_temp) & 
                                  (df_filtered['temperature_2m_mean'] <= max_temp)]
else:
    st.sidebar.write("No data available for the selected dataset.")
plot_types = ['Line', 'Scatter', 'Bar']
selected_plot_type = st.sidebar.selectbox('Select plot type:', plot_types)

# Generating the selected plot
x_col = st.sidebar.selectbox('Select X-axis variable:', df_filtered.columns)
y_col = st.sidebar.selectbox('Select Y-axis variable:', df_filtered.columns, index=df_filtered.columns.get_loc('shortwave_radiation_sum'))

def generate_plot(plot_type, df, x, y):
    if plot_type == 'Line':
        fig = px.line(df, x=x, y=y, title=f'Edited Daily Graph: {x} vs {y}')
    elif plot_type == 'Scatter':
        fig = px.scatter(df, x=x, y=y, title=f'Edited Daily Graph: {x} vs {y}')
    elif plot_type == 'Bar':
        fig = px.bar(df, x=x, y=y, title=f'Edited Daily Graph:{x} vs {y}')
    return fig

if not df_filtered.empty:
    plot = generate_plot(selected_plot_type, df_filtered, x_col, y_col)
    st.plotly_chart(plot)
else:
    st.write("No data available for plotting.")

# Reset and legend
if st.sidebar.button("Reset Filters"):
    st.experimental_rerun()
# Reset filters button


# Main panel: display the dataframe or a plot
#if 'df_filtered' in locals() and not df_filtered.empty:
 #   st.write("### Filtered Data Visualization", df_filtered)
#else:
  #  st.write("No data to display after applying filters.")


# Sidebar for DataFrame selection
st.sidebar.title("DataFrame Selection for Hourly Data")
dataframe_options = {'Provo Hourly Data': df_hourly_provo,
                     'Redmond Hourly Data': df_redmond_hour}
selected_dataframe_name = st.sidebar.selectbox("Select the dataset:", list(dataframe_options.keys()))
df_selected = dataframe_options[selected_dataframe_name]

# Updating existing filters to use selected DataFrame
st.sidebar.title("Filter Options for Hourly Data")

# Date range filter
if not df_selected.empty:
    start_date, end_date = st.sidebar.date_input(
        "Select date range for hourly data",
        [df_selected['date'].min(), df_selected['date'].max()],
        min_value=df_selected['date'].min(),
        max_value=df_selected['date'].max()
    )
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    df_filtered = df_selected[(df_selected['date'] >= start_date) & 
                                (df_selected['date'] <= end_date)]

    # Weather code filter
    weather_options = df_filtered['weather_code'].dropna().unique()
    selected_weather = st.sidebar.multiselect('Select weather conditions for hourly data', options=weather_options)
    if selected_weather:
        df_filtered = df_filtered[df_filtered['weather_code'].isin(selected_weather)]

    # Temperature range selector
    if not df_filtered.empty:
        min_temp, max_temp = st.sidebar.slider("Select temperature range (°C) for hourly data", 
                                               float(df_filtered['temperature_2m'].min()), 
                                               float(df_filtered['temperature_2m'].max()), 
                                               (float(df_filtered['temperature_2m'].min()), 
                                                float(df_filtered['temperature_2m'].max())))
        df_filtered = df_filtered[(df_filtered['temperature_2m'] >= min_temp) & 
                                  (df_filtered['temperature_2m'] <= max_temp)]

# Plot type selection for hourly data
plot_types = ['Line', 'Scatter', 'Bar']
selected_plot_type = st.sidebar.selectbox('Select plot type for hourly data:', plot_types)

# Generating the selected plot for hourly data
x_col = st.sidebar.selectbox('Select X-axis variable for hourly data:', df_filtered.columns)
y_col = st.sidebar.selectbox('Select Y-axis variable for hourly data:', df_filtered.columns, index=df_filtered.columns.get_loc('shortwave_radiation'))

def generate_plot(plot_type, df, x, y):
    if plot_type == 'Line':
        fig = px.line(df, x=x, y=y, title=f'Edited Hourly Graph: {x} vs {y}')
    elif plot_type == 'Scatter':
        fig = px.scatter(df, x=x, y=y, title=f'Edited Hourly Graph: {x} vs {y}')
    elif plot_type == 'Bar':
        fig = px.bar(df, x=x, y=y, title=f'Edited Hourly Graph: {x} vs {y}')
    return fig

if not df_filtered.empty:
    plot = generate_plot(selected_plot_type, df_filtered, x_col, y_col)
    st.plotly_chart(plot)
else:
    st.write("No data available for plotting after filters.")

# Reset filters button
if st.sidebar.button("Reset Filters for Hourly Data"):
    st.experimental_rerun()

st.sidebar.markdown("## Weather Code Legend")
weather_labels = {
        0: "Cloud development not observed",
    1: "Clouds dissolving or less developed",
    2: "State of sky on the whole unchanged",
    3: "Clouds generally forming or developing",
    51: "Drizzle, not freezing, continuous (slight at time of observation)",
    53: "Drizzle, not freezing, continuous (moderate at time of observation)",
    55: "Drizzle, not freezing, continuous (heavy (dense) at time of observation)",
    61: "Rain, not freezing, continuous (slight at time of observation)",
    63: "Rain, not freezing, continuous (moderate at time of observation)",
    65: "Rain, not freezing, continuous (heavy at time of observation)",
    71: "Continuous fall of snowflakes (slight at time of observation)",
    73: "Continuous fall of snowflakes (moderate at time of observation)",
    75: "Continuous fall of snowflakes (heavy at time of observation)"
}
weather_legend = "\n".join([f"- **{code}**: {label}" for code, label in weather_labels.items()])
st.sidebar.write(weather_legend)