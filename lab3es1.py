import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score



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

'''
data = {'A': [1, 2, 3, 4],
'B': [5, 6, 7, 8],
'C': [9, 10, 11, 12]}
df = pd.DataFrame(data)
print(df)

selected_data = df.iloc[1:3, 0:2]  seleziona riga con indice da 1(incluso) a 3(escluso),
                                    seleziona colonna con indice da 0(incluso) a 2(escluso)
print(selected_data)
'''
dataset = dataset.iloc[:, :-1] # Remove the last column from the dataset

# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Normalize the 'age' attribute
dataset['Age'] = scaler.fit_transform(dataset['Age'].values.reshape(-1, 1)) #Standardize the 'Age' column

#CLUSTERIZZAZIONE