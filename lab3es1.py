import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_excel("C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab3Materiale\\UsersSmall.xls")

#print(dataset)

dataset.replace(to_replace='?', value=np.nan, inplace=True)

#print(dataset.isnull().sum())

# Replace NaN values with the average value for numerical columns
for col in dataset.select_dtypes(include=np.number).columns:
    dataset[col].fillna(dataset[col].mean(), inplace=True) # Get the average value for the column and replace NaN values with it


# Replace NaN values with the most frequent value for non-numerical columns
for col in dataset.select_dtypes(exclude=np.number).columns:
    mode_val = dataset[col].mode()[0] # Get the most frequent string value
    dataset[col].fillna(mode_val, inplace=True) # Get the most frequent value for the column and replace NaN values with it

#print(dataset)

#condizione per l'attributo età
condition = (dataset['Age'] >= 0) & (dataset['Age'] < 105)
#applico la condizione al dataset
dataset = dataset[condition].reset_index(drop=True)




