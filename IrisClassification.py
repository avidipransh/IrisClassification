"""
Name Of Author: Avi Dipransh
Date:13/01/2019
Description: This is a model to classify Iris flowers based on the given parameters.0
"""

#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import urlopen

#scrapping data from the website
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
html = urlopen(url)
soup = BeautifulSoup(html , 'lxml')
title = soup.title
text = soup.get_text()
from io import StringIO
df = StringIO(text)
dataset = pd.read_csv(df , sep = ",")

#splitting the deopendent and independent variables
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 4].values

#splitting data into training and test sets
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

#since the values in the data are close to each ther hence feature scaling of values is not required.

#classifying the data using SVM
from sklearn.svm import SVC
classifier = SVC(kernel = "linear" , degree =1)
classifier.fit(X_train , y_train)

#predicting the results using KNN
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(metric = "minkowski" , p = 2)
classifier2.fit(X_train , y_train)

#predicting the test results for KNN
y_pred2 = classifier2.predict(X_test)

#predicing the resuls for test set for SVM.
y_pred = classifier.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
cm2 = confusion_matrix(y_test , y_pred2)

