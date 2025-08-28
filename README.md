# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries – Load Python libraries for data handling, visualization, and machine learning.
2. Load Data – Read the CSV file into a DataFrame and inspect the first and last few rows.
3. Split Features and Target – Separate the dataset into input features (X) and output target (y).
4. Split Data into Training and Testing Sets – Divide data into training and test sets for model evaluation.
5. Train Linear Regression Model – Fit a Linear Regression model using the training data.
6. Make Predictions – Predict the target values for the test data using the trained model.
7. Visualize Results – Plot actual vs predicted values for training and test sets to see the fit.
8. Evaluate Model – Calculate metrics like MSE, MAE, and RMSE to measure model accuracy.
   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:  Sanjana J
RegisterNumber:  212224230240
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df= pd.read_csv('data.csv')

df.head()
df.tail()

X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
y_pred

y_test

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
Head Values
<img width="216" height="262" alt="Screenshot 2025-08-28 111350" src="https://github.com/user-attachments/assets/f1161dac-ffdd-495a-9563-2b4a28863753" />

Tail Values
<img width="210" height="263" alt="Screenshot 2025-08-28 111354" src="https://github.com/user-attachments/assets/8e6487bb-6936-421e-b08a-c3b9c5af535c" />

X values
<img width="316" height="555" alt="Screenshot 2025-08-28 111359" src="https://github.com/user-attachments/assets/19bf2a65-818f-4de8-9fe7-11f1cd1f80ca" />

Y values
<img width="668" height="123" alt="Screenshot 2025-08-28 111409" src="https://github.com/user-attachments/assets/dd14205a-34d2-4f7e-9032-a422be1c4cfc" />

Predicted values
<img width="667" height="206" alt="Screenshot 2025-08-28 111419" src="https://github.com/user-attachments/assets/8abf2371-6584-46ae-9fe2-e6bd31733c79" />

Actual values
<img width="677" height="89" alt="Screenshot 2025-08-28 111423" src="https://github.com/user-attachments/assets/313b0c8b-f561-4c60-b4c2-6da2a84c62f1" />

Training set
<img width="690" height="686" alt="Screenshot 2025-08-28 111429" src="https://github.com/user-attachments/assets/4d8c9ec1-2a87-410e-8883-843c665b7a80" />

Testing set
<img width="668" height="669" alt="Screenshot 2025-08-28 111437" src="https://github.com/user-attachments/assets/0b07c2e4-8104-47e7-93aa-7997f4f9845a" />

MSE, MAE and RMSE
<img width="683" height="213" alt="Screenshot 2025-08-28 111443" src="https://github.com/user-attachments/assets/7e29ed24-cd06-4a4a-9d61-939d6baa6c4f" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
