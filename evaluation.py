import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from math import radians,sin,cos,sqrt,atan2
from model import *


# Define a function to preprocess and extract features from trajectory data
def process_data(traj):
    # Calculate mean and standard deviation of longitude and latitude
    longitude = traj[:,0]
    latitude = traj[:,1]
    time_diff = pd.to_datetime(traj[:,2]).values.astype(np.int64) // 10**9

    mean_longitude = np.mean(longitude)
    mean_latitude = np.mean(latitude)
    standard_longitude = np.std(longitude)
    standard_latitude = np.std(latitude)
    
    distance = distance_travelled(longitude,latitude)
    mean_speed = distance/np.sum(time_diff)

    feature = [
        mean_longitude,
        mean_latitude,
        standard_longitude,
        standard_latitude,
        mean_speed
    ]
    return feature

def distance_travelled(longitude, latitude):
    R = 6371.0
    distance = 0.0
    for i in range(1,len(longitude)):
    # Convert latitude and longitude from degrees to radians
        lat1_rad = radians(latitude[i-1])
        lon1_rad = radians(longitude[i-1])
        lat2_rad = radians(latitude[i])
        lon2_rad = radians(longitude[i])

        # Differences in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Calculate the distance in kilometers
        distance = R * c
    return distance


def run(data, model):
    # Ensure data is a NumPy array
    data = np.array(data)
    data = data.reshape(1,-1)
    # # Transform input data using the previously fitted scaler
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Make predictions
    predictions = model.predict(np.array([data])).mean()
    return predictions







