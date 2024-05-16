import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix



dataset = pd.read_excel("C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab5Materiale\\user.xlsx")

#print(dataset)
#rinomino la colonna Response in Label
dataset = dataset.rename(columns={'Response': 'Label'})
#print(dataset)

# Split the dataset into features (X) and target variable (y)
X = dataset.drop(columns=['Label']) # Features
y = dataset['Label'] # Target variable
# Label encoding  --> Assegna integer uguali a text uguali
labelencoder = LabelEncoder()
# Apply label encoding to each column, except for the age column
for column in X.columns:
    if column != 'Age': X[column] = labelencoder.fit_transform(X[column])

#print(X)

#inizializzazione del decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=20,min_impurity_decrease=0.001)
# Train the Decision Tree Classifier
clf.fit(X, y)

tree_structure = export_text(clf, feature_names=list(X.columns))
#print(tree_structure)




new_dataset = pd.read_excel("C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab5Materiale\\prospect.xlsx")

#encoding sul nuovo dataset
X = new_dataset
# Label encoding for the new_dataset
labelencoder = LabelEncoder()
# Apply label encoding to each column, except for the age column
for column in X.columns:
    if column != 'Age': X[column] = labelencoder.fit_transform(X[column])

# Predict the target variable of the new dataset
y_pred = clf.predict(X)
# print the prediction
print(y_pred)

