import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix


dataset = pd.read_excel("C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab6Materiale\\user.xlsx")

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

# Transform Negative into 0value and Positive into 1 value (use label encoder with .fit_transform)
y = labelencoder.fit_transform(y)
#print(y)


# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Create a Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=20, max_depth=3,random_state=42)
# Train the model using the training sets
random_forest.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = random_forest.predict(X_test)
# Evaluate the model: Accuracy, Precision, Recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Print the evaluation metrics
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
