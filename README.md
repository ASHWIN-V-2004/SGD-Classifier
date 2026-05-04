# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize dataset, normalize features, and set weights (θ) to zero.
2. Compute predicted probabilities using the sigmoid function.
3. Update weights iteratively using gradient descent to minimize error.
4.Predict classes using a threshold (0.5) and evaluate accuracy. 

## Program:
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: Ashwin V
RegisterNumber:212225040034
 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data (1).csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values


scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))


theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show() 

```

## Output:
<img width="752" height="601" alt="image" src="https://github.com/user-attachments/assets/20896af5-32d2-4f70-90dd-3b3bfdc67459" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
