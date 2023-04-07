# Heart Disease Classifier

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('C:\\HvA\\bIG DATA\\pro classifier\\heart-disease-classifier\\Heart Disease Dataset.csv')

# Split data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict the output for test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# Plot the results
plt.bar(['Training Set', 'Test Set'], [len(X_train), len(X_test)], color=['blue', 'green'])
plt.title('Heart Disease Classifier Dataset Split')
plt.xlabel('Dataset')
plt.ylabel('Number of Samples')
plt.show()

plt.bar(['Accuracy'], [accuracy], color='red')
plt.title('Heart Disease Classifier Accuracy')
plt.xlabel('Metrics')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()