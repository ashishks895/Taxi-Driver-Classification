# Taxi-Driver-Classification
Taxi Trajectory Classification using Deep Learning

The goal of this project is to classify taxi drivers based on their driving patterns using trajectory data. 
This classification is essential for various applications, such as driver behavior analysis, anomaly detection, and personalized services. 
We propose a methodology for processing the data, generating relevant features, designing neural network structures, and evaluating the 
models. 
We aim to achieve a classification accuracy of at least 60%. 



 Network Structure 
 
We explore two different neural network structures: 

Simple RNN Model: 
• Two Simple RNN layers with 64 and 32 units, respectively. 
• A Dense layer with 32 units and ReLU activation. 
• A Dropout layer with 20% dropout rate. 
• A Dense output layer with 5 units and sigmoid activation. 

LSTM Neural Network: 
• Two LSTM layers with 64 and 32 units, respectively. 
• A Dense layer with 32 units and ReLU activation. 
• A Dropout layer with 20% dropout rate. 
• A Dense output layer with 5 units and sigmoid activation. 

Training & Validation Process 
• We use the Adam optimizer with custom learning rates for both models. 
• Models are compiled with sparse categorical cross-entropy loss and accuracy metric. 
• Training is performed for 50 epochs with a batch size of 16. 
• We save the trained models for evaluation. 
