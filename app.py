#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


import pandas as pd



# In[3]:


import numpy as np


# In[4]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy import stats


# In[5]:


# Load data
filepath = "C:\\Users\\chakri\\Downloads\\energy_production.csv"
df = pd.read_csv(filepath, sep=';')


# In[6]:


# Remove outliers using Z-score
z_scores_r_humidity = np.abs(stats.zscore(df['r_humidity']))
z_scores_amb_pressure = np.abs(stats.zscore(df['amb_pressure']))
threshold = 3
df_cleaned = df[(z_scores_r_humidity < threshold) & (z_scores_amb_pressure < threshold)]



# In[7]:


# Prepare data
X = df_cleaned[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df_cleaned['energy_production']



# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Train XGBoost model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)



# In[10]:


# Streamlit app
st.title('Energy Production Prediction')



# In[11]:


# Sidebar for user input
st.sidebar.title('Input Parameters')
temperature = st.sidebar.slider('Temperature', min_value=df['temperature'].min(), max_value=df['temperature'].max())
exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum', min_value=df['exhaust_vacuum'].min(), max_value=df['exhaust_vacuum'].max())
amb_pressure = st.sidebar.slider('Ambient Pressure', min_value=df['amb_pressure'].min(), max_value=df['amb_pressure'].max())
r_humidity = st.sidebar.slider('Relative Humidity', min_value=df['r_humidity'].min(), max_value=df['r_humidity'].max())



# In[12]:


# Predict button
if st.sidebar.button('Predict'):
    # Predict energy production
    input_data = np.array([[temperature, exhaust_vacuum, amb_pressure, r_humidity]])
    prediction = xgb_model.predict(input_data)[0]
    st.write(f'Predicted Energy Production: {prediction:.2f} MW')

