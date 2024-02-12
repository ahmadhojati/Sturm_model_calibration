#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Add libraries
import numpy as np
import pandas as pd
from datetime import date
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings


# In[2]:


# Define a function to determine the water year start date based on the year
def determine_water_year_start(year):
    # You can customize this logic to determine the water year start based on your needs.
    # For this example, we'll assume a water year starts on October 1st for all years.
    return pd.to_datetime(f'{year}-10-01')


# In[3]:


def DOY_(Y, M, D):
    # Determine day of year
    # −92 (1 October) to +181 (30 June)
    DOY = date.toordinal(date(Y, M, D)) - date.toordinal(date(Y, 12, 31)) - 1
    if DOY <= -184:
        DOY = DOY + 366
    return DOY


# In[20]:


# Load the pandas DataFrame from the pkl file
df = pd.read_csv('data/Sentinel_SNOTEL_Soil_Moisture_Landsat8_Climate_class_2014_2023_10m_100m_300m.csv')


# In[140]:


res = 10                     # process resolution
min_density = 50              # minimum snow density to process
max_density = 550             # maximum snow density to process
min_snow_depth = 30           # minimum snow depth to process
soil_depth = 5                # soil depth for soil moisture
Climate_class = 2             # climate class for processing
max_TAVG = 72                 # maximum average air temperature fro processing
thr = 1                       # minimum length of time series at each point 
threshold = 10                # minimum number of dates within a year
orbit = np.nan                     # 0 for ascending (dry snow) and 1 for descending (wet snow)
if orbit == 0:
    name = 'Ascending'        # prefix for saving models and figures
elif orbit == 1:
    name = 'Descending'
else:
    name = ''
    
#temp = '_without_TAVG_Prep'
temp = ''


# Model training parameters
epochs = 1500                 # deep learning number of epochs for training
batch_size = 32               # batch size for training
lr = 0.001                    # learning rate for model compile


# In[141]:


# ## Apply required filters

# Snow density above 50 and below 550
df_subset = df[(df['Density'] >= min_density) & (df['Density'] <= max_density)]

# # Remove average temperature above zero
df_subset = df_subset[df_subset['TAVG']<=max_TAVG]      # 0 degree centigerade
# df_subset = df_subset[df_subset['TAVG'] <= max_TAVG]  # remove above 72 F

# # Snow depth above 30 cm
df_subset = df_subset[df_subset['SD'] > min_snow_depth]

# # Filter based on resolution
df_subset = df_subset[df_subset['resolution'] == res]

# Top level soil moisture 
df_subset = df_subset[df_subset['depth'].isin([soil_depth, np.nan])]

# Replace SMS = 0 with NAs
df_subset['SMS'] = df_subset['SMS'].replace(0, np.nan)

# ### Add orbit information to the data

# **1 means descending and 0 means ascending orbits**

df_subset['Date'] = pd.to_datetime(df_subset['Date'])

df_subset['Time'] = pd.to_datetime(df_subset['Time'])

# Extract the hour from the "Time" column
df_subset['Hour'] = df_subset['Time'].apply(lambda x: x.hour)

# Create the "Orbit" column based on the hour
df_subset['Orbit'] = (df_subset['Hour'] >= 12).astype(int)

# Drop the intermediate "Hour" column if not needed
df_subset = df_subset.drop(columns=['Hour'])
df_subset = df_subset.drop(columns=['Time'])
df_subset = df_subset.drop(columns=['Sensor_ID'])      
df_subset = df_subset.groupby(['Coordinate','Date']).mean().reset_index()


# ## Find the coordinates with at least 200 measurements

# Minimum length of time series at each point
# thr = 200

# Find the coordinates with at least 200 measurements
mask = (df_subset.groupby('Coordinate').count()>thr).reset_index()['Date']
sel_coord = (df_subset.groupby('Coordinate').count()>thr).reset_index()['Coordinate'][mask]

# Filter the df by selected coordinates
selected_rows = df_subset[df_subset['Coordinate'].isin(sel_coord)]

data = selected_rows.loc[:,['Density','Coordinate','Date','SD','sigma_0_VV','sigma_0_VH','Inc',
                            'S_Elevation','PRCPSA','TAVG', 'Climate_class','Orbit','SMS']]


# ## Modeling and Forecasting
data['DOY'] = None


# Apply the function to create a new column for the start of the water year
water_year_start = pd.Series((data['Date'].dt.year-1)).apply(determine_water_year_start)

# Compute the day of the water year
data['DOY'][:] = (pd.Series(data['Date']) - water_year_start).dt.days + 1

# Change the type
data['DOY'] = data['DOY'].astype('int32')

data.loc[data['DOY'] > 365, 'DOY'] -= 365

# ### Lets filter the data to only rows where there are at least 10 dates in each year
# Add a year column to the datafram
data['Year'] =  data['Date'].dt.year

# Make sure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Group by 'Coordinate' and 'Year', count the number of dates in each group
grouped_counts = data.groupby(['Coordinate', 'Year'])['Date'].count().reset_index(name='Count')

# Identify the rows where the count is below the threshold
below_threshold_rows = grouped_counts[grouped_counts['Count'] <= threshold]

# Filter out rows where the count is below the threshold
mask = data[['Coordinate', 'Year']].apply(tuple, axis=1).isin(below_threshold_rows[['Coordinate', 'Year']].apply(tuple, axis=1))
data =  data[~mask]

# Drop Year column
data.drop(columns = 'Year', inplace = True)

# Replace 'inf' with NaN and then drop NaN values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop na from the dataframe
data.dropna(inplace = True)


# In[142]:


# Set seeds for reproducibility
np.random.seed(42)
#tf.set_random_seed(42)
tf.random.set_seed(42)

data_subset = data.copy()
data_subset = data_subset[data_subset['Climate_class']==Climate_class]
if not np.isnan(orbit):
    data_subset = data_subset[data_subset['Orbit']==orbit]   # Ascending = 0, Descending = 1

X = data_subset.iloc[:,[3,4,5,6,8,9,13]].reset_index()     # With TAVG and Precipitation
#X = data_subset.iloc[:,[3,4,5,6,13]]     # Without TAVG and Precipitation
y = data_subset['Density']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[143]:


Y = list(data_subset['Date'].dt.year)
M = list(data_subset['Date'].dt.month)
D = list(data_subset['Date'].dt.day)

DOY = np.zeros(len(Y))*np.nan
for i in np.arange(len(Y)):
    DOY0 = DOY_(Y[i],M[i],D[i])
    DOY[i] = DOY0


# In[149]:


if Climate_class == 1:
    Climate = 'Tundra'
elif Climate_class == 3:
    Climate = 'Maritime'
elif Climate_class == 5:
    Climate = 'Prairie'
elif Climate_class == 2:
    Climate = 'Alpine'
else:
    Climate = ''


# In[150]:


# Sturm's model coefficients
# Values in this table are from https://doi.org/10.1175/2010JHM1202.1
data = {'Climate class': ['Alpine', 'Maritime', 'Prairie', 'Tundra', 'Taiga'],
        'max_density': [0.5975, 0.5979, 0.5940, 0.3630, 0.2170],
        'init_density': [0.2237, 0.2578, 0.2332, 0.2425, 0.2170],
        'k1': [0.0012, 0.0010, 0.0016, 0.0029, 0.0000],
        'k2': [0.0038, 0.0038, 0.0031, 0.0049, 0.0000]}

# ## Calibrated coeffecients
# data = {'Climate class': ['Alpine', 'Maritime', 'Prairie', 'Tundra', 'Taiga'],
#         'max_density': [0.5975, 0.5979, 0.5940, 0.58079877, 0.2170],
#         'init_density': [0.19616258, 0.30306748, 0.17085892, 0.21717764, 0.2170],
#         'k1': [0.00166234, 0.00082006, 0.00284942, 0.00253816, 0.0000],
#         'k2': [0.0041464, 0.00501446, 0.00535967, 0.00357082, 0.0000]}

# Create a DataFrame from the coefficients data
coef = pd.DataFrame(data)

# coefficients from the table 
#Maximum bulk density
max_density = coef[coef['Climate class'] ==Climate]['max_density'].values

#Initial density
init_density = coef[coef['Climate class'] ==Climate]['init_density'].values

#Densification parameter for depth
k1 = coef[coef['Climate class'] ==Climate]['k1'].values

#Densification parameters for day of year
k2 = coef[coef['Climate class'] ==Climate]['k2'].values



# In[151]:


DOY_train = DOY[X_train.index]
H_train = X_train['SD']


# In[152]:


import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Generate synthetic data for illustration purposes (replace with your actual data)
np.random.seed(42)


# Objective function
def objective_function(coefficients, y_train,H,DOY ):
    k1, k2, max_density, init_density = coefficients

    predictions = (max_density - init_density) * (1 - np.exp(-k1 * np.array(H) - k2 * DOY)) + init_density
    return mean_squared_error(y_train,predictions,squared=False)

# Initial coefficients
initial_coefficients = [k1, k2,max_density,init_density]


# In[153]:


# Example bounds for max_density and init_density
bounds_max_density = (0.4*max_density, 1*max_density)  # Replace with your desired bounds
bounds_init_density = (0.4*init_density, 1.6*init_density)  # Replace with your desired bounds

# Combine bounds for all parameters
bounds = [(None, None), (None, None), bounds_max_density, bounds_init_density]

# Minimize the objective function with bounds
result = minimize(objective_function, initial_coefficients, args=(y_train/1000, H_train, DOY_train),
                  method='Nelder-Mead', bounds=bounds)

# Calibrated coefficients
calibrated_k1, calibrated_k2, calibrated_max_density, calibrated_init_density  = result.x


# In[155]:


result.x


# In[156]:


# Sturm function
def sturm(coefficients, y_train,H,DOY ):
    k1, k2, max_density, init_density = coefficients

    predictions = (max_density - init_density) * (1 - np.exp(-k1 * np.array(H) - k2 * DOY)) + init_density
    return predictions


# #### Train RMSE

# In[157]:


coefficients = result.x
objective_function(coefficients, y_train/1000,H_train,DOY_train )


# #### Train R2

# In[158]:


y_pred = sturm(coefficients, y_train/1000,H_train,DOY_train )

r2_score(y_train/1000,y_pred)


# #### Test RMSE

# In[159]:


DOY_test = DOY[X_test.index]
H_test = X_test['SD']
objective_function(coefficients, y_test/1000,H_test,DOY_test )


# ### Test R2

# In[160]:


y_pred = sturm(coefficients, y_test/1000,H_test,DOY_test )

r2_score(y_test/1000,y_pred)


# In[ ]:




