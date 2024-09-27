# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Start 
2. Load the salary dataset into a Pandas DataFrame and inspect the first few rows using data.head().
3: Check the dataset for missing values using data.isnull().sum() and inspect the data structure using data.info().
4: Preprocess the categorical data. Use LabelEncoder to convert the "Position" column into numerical values.
5: Define the feature matrix (X) by selecting the relevant columns (e.g., Position, Level), and set the target variable (Y) as the "Salary" column.
6: Split the dataset into training and testing sets using train_test_split() with a test size of 20%.
7: Initialize the Decision Tree Regressor and fit the model to the training data (x_train, y_train).
8: Predict the target values on the testing set (x_test) using dt.predict().
9: Calculate the Mean Squared Error (MSE) using metrics.mean_squared_error() and the R-squared score (r2_score()) to evaluate the model's performance.
10: Use the trained model to predict the salary of an employee with specific input features (dt.predict([[5,6]])).
11: End
```

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Prashanth.K
RegisterNumber:  212223230152

```
```
import pandas as pd
data = pd.read_csv(r"C:\Users\admin\Desktop\Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse
r2 = metrics.r2_score(y_test, y_pred)
r2
dt.predict([[5, 6]])
```

## Output:

## Head:
![image](https://github.com/user-attachments/assets/39868106-da78-4346-a1ec-086baae561ae)

## MSE:
![image](https://github.com/user-attachments/assets/2170ef0e-c296-4053-b460-5abb6fa8aedd)

## R2:
![image](https://github.com/user-attachments/assets/1f666d3b-f16c-4f93-a0c7-d6a8d81c16a3)

## Prediction:
![image](https://github.com/user-attachments/assets/4314d207-755d-410d-b332-54f2896ba982)







## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
