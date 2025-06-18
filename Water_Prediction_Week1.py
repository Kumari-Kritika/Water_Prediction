#!/usr/bin/env python
# coding: utf-8

# In[2]:


##########I USED EMOJIS TO MAKE IT MORE CREATIVE AND INTERESTING #######
import pandas as pd           # 🐼 For dataframes and table magic
import numpy as np            # 🔢 For mathematical power

from sklearn.multioutput import MultiOutputRegressor  # 🎯 Predict multiple outputs
from sklearn.ensemble import RandomForestRegressor     # 🌳 Ensemble learning magic
from sklearn.model_selection import train_test_split   # ✂️ Split for training and testing
from sklearn.metrics import mean_squared_error, r2_score  # 📏 For measuring model performance


# In[3]:


# Read CSV with semicolon separator
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')

# ✅ Confirm successful load
print("✅ Dataset loaded successfully!")



# In[4]:


# View basic info about data (types, nulls, etc.)
df.info()  # ℹ️ Quick glance at data structure

# Print dataset shape
print(f"📐 Rows and columns: {df.shape}")



# In[5]:


# Columns of interest 🔎
columns = ['d', 'date', 'NH4', 'BSK5', 'Suspended', 'O2', 
           'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# 📋 Summary statistics for selected columns
# load the dataset
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
df


# In[6]:


# 🧠 Structure again
df.info()

# Shape again for verification
print(f"🧾 Rows x Columns: {df.shape}")

# 📈 Stats overview
print(df.describe().T)

# ❓ Missing values?
print("🧹 Missing values per column:\n", df.isnull().sum())


# In[7]:


# Convert date string to datetime object
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

# ✅ Check
df.info()


# In[8]:


# Sort by id and date 🗂️
df = df.sort_values(by=['id', 'date'])
print("🔃 Sorted by ID and Date:")
print(df.head())

# Extract year and month 📆
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

print("🗓️ Year and Month extracted:")
print(df.head())


# In[ ]:


# 🎯 Target pollutants to predict
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

