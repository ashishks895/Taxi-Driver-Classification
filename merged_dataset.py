import os
import pandas as pd
import numpy as np

# Specify the directory containing CSV files
csv_directory = 'Specify the directory containing CSV file'

dataframe = []

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)
        dataframe.append(df)

merged_data = pd.concat(dataframe, ignore_index=True)

merged_data.to_csv('merged_data.csv',index=False)