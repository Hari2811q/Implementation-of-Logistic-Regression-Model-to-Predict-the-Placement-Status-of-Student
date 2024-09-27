# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Hariprasath.R
RegisterNumber: 212223040059
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
PLACEMENT DATA:

![image](https://github.com/user-attachments/assets/652621df-9e68-4282-8905-f7569f535cba)

SALARY DATA:

![image](https://github.com/user-attachments/assets/0d171efa-4cb8-453d-aa00-38469f04b765)

CHECKING THE NULL() FUNCTION:]

![image](https://github.com/user-attachments/assets/5a5f17a9-65ee-4af2-93ea-77c6c26e6db5)

DATA DUPLICATE:

![image](https://github.com/user-attachments/assets/a7e9885c-a20a-4c5b-9011-a6399c018c72)

PRINT DATA:

![image](https://github.com/user-attachments/assets/928e236a-c490-4389-ad75-36b9c1d3f42b)

DATA_STATUS:

![image](https://github.com/user-attachments/assets/4479af32-98f8-48e1-b423-5565c11a2423)

DATA_STATUS:

![image](https://github.com/user-attachments/assets/f68c871e-daa9-420a-a81e-3026b4a517e5)

Y_PREDICTION ARRAY:

![image](https://github.com/user-attachments/assets/f8950889-f6e6-440c-aa15-75bd56f34ba2)

ACCURACY VALUE:

![image](https://github.com/user-attachments/assets/071c639a-c319-43b7-8f5a-225c36f32919)

CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/82ae29d8-db98-4615-aec6-d8fd57172228)

CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/9e1f44e9-eb1c-4821-905e-8aaf611b284b)

PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/b4ffa7ab-409c-4244-bcad-791d554e9d6c)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
