#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn  
import matplotlib.pyplot as plt
import time
import datetime as dt


# In[2]:


data = pd.read_csv('trip_data.csv')
print("Total number of dataset : ", data.shape[0])


# In[3]:


## shows sample 5 data randomly
data.head(5)


# In[4]:


## Shows the statistics of the dataset
data.describe()


# In[5]:


## Checking for nan values
data.isnull().sum()


# **Now after checking the nan values we will check the data types of all the columns as we need numerical values 
# for machine learning algorithms we will convert the all the data into numerical data.**

# In[6]:


## Shows the datatypes of the data.
data.info()


# In[7]:


data['Date'] = pd.to_datetime(data['Date'])
data['Beginning Time'] = pd.to_datetime(data['Beginning Time'])
data['End Time'] = pd.to_datetime(data['End Time'])


# In[8]:


data['total_time']= data['End Time'] - data['Beginning Time']
data['total_time']=data['total_time']/np.timedelta64(1,'m')


# In[9]:


data.sample(3)


# In[10]:


data['total_time']= data['End Time'] - data['Beginning Time']
data['total_time']=data['total_time']/np.timedelta64(1,'h')
data['Avg_Speed'] = data['Mileage']/data['total_time']


# In[11]:


data['Month'] = data['Date'].dt.month
data['Beginning Time'] = data['Beginning Time'].dt.hour
data['End Time'] = data['End Time'].dt.hour


# In[12]:


data['Date'] = data['Date'].dt.weekday


# In[13]:


data.rename(columns={'Date':'DayofWeek'}, inplace= True)


# In[14]:


data.info()


# In[15]:


data['peak/off peak'].value_counts()


# In[16]:


data1 = {"peak/off peak": {"Off peak": 0, "peak": 1, "peak ":1}}


# In[17]:


data = data.replace(data1)
data.head()


# In[18]:


data.info()


# In[19]:


y= data['End Time']


# In[20]:


X= data[['DayofWeek','peak/off peak','Beginning Time','Mileage','Initial latitude ','Initial longitude','Final latitude','Final longitude','total_time','Avg_Speed','Month']]


# In[21]:


X.shape


# In[22]:


fig = plt.figure(figsize = (10, 10))
sns.heatmap(X.corr(), annot = True)


# In[23]:


data['Initial latitude '].min()


# In[24]:


data['Initial latitude '].max()


# In[25]:


data['Final latitude'].min()


# In[26]:


data['Final latitude'].max()


# In[27]:


data['Initial longitude'].min()


# In[28]:


data['Initial longitude'].max()


# In[29]:


data['Final longitude'].min()


# In[30]:


data['Final longitude'].max()


# In[31]:


data


# In[32]:


long=(38.668259,38.990913)
lat=(8.74698,9.07393)
data.plot(kind='scatter',x='Final latitude',y='Final longitude',color='blue',s=0.9,alpha=0.6)
plt.title('Destination')
plt.ylim(long)
plt.xlim(lat)


# In[33]:


pip install folium


# In[34]:


import folium
start_location=folium.Map(location=[8.74698,38.668259],tiles='openStreetMap',zoom_start=12)
for each in data[:10000].iterrows():
    folium.CircleMarker([each[1]['Final latitude'],each[1]['Final longitude']],
                       radius=3,color='blue',
                       popup=str(each[1]['Final latitude'])+","+str(each[1]['Final longitude']),
                       fill_color='#FD8A6C').add_to(start_location)


start_location   


# In[ ]:


import folium
end_location=folium.Map(location=[8.747010000000001,38.668251],tiles='openStreetMap',zoom_start=12)
for each in data[:10000].iterrows():
    folium.CircleMarker([each[1]['Initial latitude '],each[1]['Initial longitude']],
                       radius=3,color='red',
                       popup=str(each[1]['Initial latitude '])+","+str(each[1]['Initial longitude']),
                       fill_color='#FD8A6C').add_to(end_location)


end_location   


# In[35]:


# vectorized haversine function
from haversine import haversine
from haversine import haversine_vector, Unit
from math import sin, cos, sqrt, atan2, radians


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    lat1 = data['Initial latitude']
    lon1 = data['Initial longitude']
    lat2 = data['Final latitude']
    lon2 = data['Final longitude']
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

   # a = np.sin((lat2-lat1)/2.0)**2 + \
   #     np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    data['distance'] = haversine_vector([lat1, lon1], [lat2, lon2], Unit.KILOMETERS)


# In[ ]:


data.head()


# In[ ]:


data['distance'].isnull().sum()


# In[ ]:


data['distance'].dropna(inplace=True)


# In[ ]:


data.sample(5)


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')

data.hist(bins=20, figsize=(20,15))
plt.savefig('histogram')
plt.show()


# In[3]:


### Save the prepared dataset
data.to_csv('final_data.csv',index=False)


# In[3]:


data = pd.read_csv('final_data.csv')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[5]:


df = pd.read_csv("final_data.csv")
df.head()


# In[20]:


df.columns


# In[6]:


df.columns = [c.replace(' ', '_') for c in df.columns]


# In[7]:


df.columns


# In[8]:


feature_columns = [
    'DayofWeek', 'peak/off_peak', 'Beginning_Time', 'Mileage',
       'Initial_latitude_', 'Initial_longitude', 'Final_latitude',
       'Final_longitude', 'total_time', 'Avg_Speed', 'Month', 'distance'
]


# In[9]:


df = df.fillna(0)


# In[10]:


df.info()


# In[11]:


import sklearn  
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X=scalar.fit_transform(X)


# In[11]:


from sklearn.model_selection import train_test_split


X = df[feature_columns]
y = df.End_Time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[31]:


from sklearn.ensemble import RandomForestRegressor
import time
start_time=time.time()
RF=RandomForestRegressor()
RF.fit(X_train,y_train)
pre=RF.predict(X_test)
print("execution time="+str((time.time()-start_time))+'sec')


# In[32]:


from sklearn import metrics
print("MAE=",metrics.mean_absolute_error(y_test,pre))
print("MSE=",metrics.mean_squared_error(y_test,pre))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,pre)))
print("R2 Score=",RF.score(X_train,y_train))


# In[ ]:


X_over=X
y_over=y


# In[13]:


from sklearn.ensemble import GradientBoostingRegressor
import time
start_time=time.time()
GB=GradientBoostingRegressor()
GB.fit(X_train,y_train)
pre=GB.predict(X_test)
print("execution time="+str((time.time()-start_time))+'sec')
from sklearn import metrics
print("MAE=",metrics.mean_absolute_error(y_test,pre))
print("MSE=",metrics.mean_squared_error(y_test,pre))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,pre)))
print("R2 Score=",GB.score(X_train,y_train))


# In[15]:


from sklearn.linear_model import LinearRegression
import time
start_time=time.time()
LR=LinearRegression().fit(X_train,y_train)
pre=LR.predict(X_test)
print("execution time="+str((time.time()-start_time))+'sec')
from sklearn import metrics
print("MAE=",metrics.mean_absolute_error(y_test,pre))
print("MSE=",metrics.mean_squared_error(y_test,pre))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,pre)))
print("R2 Score=",LR.score(X_train,y_train))


# In[16]:


from sklearn import linear_model
import time
start_time=time.time()
LM=linear_model.Lasso(alpha=0.1)
LM.fit(X_train,y_train)
pre=LM.predict(X_test)
print("execution time="+str((time.time()-start_time))+'sec')
from sklearn import metrics
print("MAE=",metrics.mean_absolute_error(y_test,pre))
print("MSE=",metrics.mean_squared_error(y_test,pre))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,pre)))
print("R2 Score=",LM.score(X_train,y_train))


# In[17]:


from sklearn.neighbors import KNeighborsRegressor
import time
start_time=time.time()
KNN=KNeighborsRegressor(n_neighbors=5)
KNN.fit(X_train,y_train)
pre=KNN.predict(X_test)
print("execution time="+str((time.time()-start_time))+'sec')
from sklearn import metrics
print("MAE=",metrics.mean_absolute_error(y_test,pre))
print("MSE=",metrics.mean_squared_error(y_test,pre))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,pre)))
print("R2 Score=",KNN.score(X_train,y_train))


# In[19]:


from sklearn.svm import SVR
import time
start_time=time.time()
SV=SVR(kernel='rbf')
SV.fit(X_train,y_train)
pre=SV.predict(X_test)
print("execution time="+str((time.time()-start_time))+'sec')
from sklearn import metrics
print("MAE=",metrics.mean_absolute_error(y_test,pre))
print("MSE=",metrics.mean_squared_error(y_test,pre))
print("RMSE=",np.sqrt(metrics.mean_squared_error(y_test,pre)))
print("R2 Score=",SV.score(X_train,y_train))


# In[ ]:


# Initialize KFold with 5 folds
kf = KFold(n_splits=5, shuffle=True)

# Lists to store performance metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
metrics = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X_over):
    X_train_over, X_test_over = X_over[train_index], X_over[test_index]
    y_train_over, y_test_over = y_over[train_index], y_over[test_index]

    # Train RF_model classifier
    RF_model.fit(X_train_over, y_train_over)

    # Make predictions
    y_pred = RF_model.predict(X_test_over)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test_over, y_pred)
    precision = precision_score(y_test_over, y_pred, average='macro')
    recall = recall_score(y_test_over, y_pred, average='macro')
    f1 = f1_score(y_test_over, y_pred, average='macro')

    # Append metrics to respective lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Print metrics for each fold
    print(f"Fold Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

