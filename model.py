import pickle
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.discriminant_analysis import StandardScaler
from math import radians,sin,cos,sqrt,atan2

data = pd.read_csv('merged_data.csv')

data['time'] = pd.to_datetime(data['time'])

# def calculate_duration(data):
#     # Extract timestamps from the trajectory
#     timestamps = data['time']

#     # Calculate the time difference between the first and last timestamp
#     duration = (timestamps[-1] - timestamps[0])

#     return duration

    
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
    
    # duration = calculate_duration(data)
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


# Initialize lists for training data and labels
def aggr(data):
    traj_raw = data.values[:,1:]
    traj = np.array(sorted(traj_raw,key = lambda d:d[2]))
    label = data.iloc[0][0]
    return [traj,label]

processed_data = data.groupby('plate').apply(aggr)


# Generate some features
training = []
labels = []
for data in processed_data:
    feature = process_data(data[0])
    label = data[1]
    training.append(feature)
    labels.append(label)

 # Normalizing the training data for Model
scaler = StandardScaler()
training = scaler.fit_transform(training)

labels = np.array(labels)
labels = labels.reshape(-1,1)
print(labels.shape)

# Reshape training data for RNN input
training = np.array(training).reshape(len(training),1,len(training[0]))
print(training.shape)
print(training.dtype)


## Simple_RNN Model

# Define your RNN model
model = Sequential()
model.add(SimpleRNN(64,  return_sequences=True,  input_shape=(1, 5)))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dense(32, activation='relu',input_dim=len(training[0])))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))

# Define your custom learning rate
custom_learning_rate = 0.001  

# Create the Adam optimizer with the custom learning rate
adam_optimizer = Adam(learning_rate=custom_learning_rate)

# Compile your model with the custom learning rate optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
model.fit(training, labels, epochs=50, batch_size=16)
pickle.dump(model,open('deep_model.pkl','wb'))


### LSTM Neural Network

# Define your RNN model
model2 = Sequential()

# LSTM layer with 64 units and return sequences
model2.add(LSTM(64, return_sequences=True, input_shape=(1, 5)))
model2.add(LSTM(32, return_sequences=False))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(5, activation='sigmoid'))

# Compile the model with custom learning rate
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(training, labels, epochs=50, batch_size=16)
pickle.dump(model2,open('deep_model.pkl','wb'))





