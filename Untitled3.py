#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the Iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None,
                    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Convert the categorical species column to numerical labels
iris['species'] = pd.factorize(iris['species'])[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris[['sepal_length', 'sepal_width']],
    iris['species'],
    test_size=0.2,
    random_state=42
)

# Create a support vector machine classifier
svm = SVC()

# Fit the classifier to the training data
svm.fit(X_train, y_train)

# Predict the species of the test data
y_pred = svm.predict(X_test)

# Plot the training data
plt.scatter(X_train['sepal_length'], X_train['sepal_width'], c=y_train)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Training Data')
plt.show()

# Plot the test data and predicted labels
plt.scatter(X_test['sepal_length'], X_test['sepal_width'], c=y_pred)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Test Data - Predicted')
plt.show()

# Plot the test data and true labels
plt.scatter(X_test['sepal_length'], X_test['sepal_width'], c=y_test)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Test Data - True')
plt.show()


# In[ ]:




