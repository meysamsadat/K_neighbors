import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

wine = load_wine()
x = wine.data
y = wine.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)

neighbors = np.arange(1,30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i ,k in enumerate(neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train,y_train)
    train_accuracy[i] = knn_model.score(x_train,y_train)
    test_accuracy[i] = knn_model.score(x_test,y_test)

plt.plot(neighbors,train_accuracy,label='train accuracy')
plt.plot(neighbors,test_accuracy,label='test accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('accuracy')






