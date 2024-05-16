import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_excel("C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab1Materiale\\UsersSmall.xls")

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
#print(dataset.isnull().sum())

#Faccio un plot per trovare gli outlier 

"""
 # Fix the 'Age' attribute on the y-axis
age_values = dataset['Age']
# Plot scatter/bubble plot with an attribute on the x-axis. Ypu caan choose what ever attribute you want
attribute = "Workclass"
plt.figure(figsize=(10, 6))
plt.scatter(dataset[attribute], age_values, s=50, alpha=0.5, label=attribute)
plt.xlabel(attribute, fontsize=10) # Adjusted x-axis label font size
plt.ylabel('Age')
plt.title('Scatter/Bubble Plot with Age on Y-axis')
plt.legend()
plt.show()
"""

#condizione per l'attributo età
condition = (dataset['Age'] >= 0) & (dataset['Age'] < 105)
#applico la condizione al dataset
dataset = dataset[condition]


 # Fix the 'Age' attribute on the y-axis
age_values = dataset['Age']
# Plot scatter/bubble plot with an attribute on the x-axis. Ypu caan choose what ever attribute you want
attribute = "Workclass"
plt.figure(figsize=(10, 6))
plt.scatter(dataset[attribute], age_values, s=50, alpha=0.5, label=attribute)
plt.xlabel(attribute, fontsize=10) # Adjusted x-axis label font size
plt.ylabel('Age')
plt.title('Scatter/Bubble Plot with Age on Y-axis')
plt.legend()
plt.show() # non ho più i dati con età fuori dalla mia condizione

#DISCRETIZAZZIONE DEI DATI  dati continui--> dati discreti

# Define bin edges
bin_edges = [0, 18, 30, 40, 50, float('inf')] # Define your own bin edges as needed
# Define bin labels
bin_labels = ['0-18', '19-30', '31-40', '41-50', '51+'] # Define labels for each bin
# Discretize 'Age' attribute using cut() function
dataset['Age'] = pd.cut(dataset['Age'], bins=bin_edges, labels=bin_labels, right=False)#right=False intervalli di destra chiusi

print(dataset)


