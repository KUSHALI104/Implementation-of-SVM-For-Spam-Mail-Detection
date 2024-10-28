# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import the required packages.

Step 2: Import the dataset to operate on.

Step 3: Split the dataset.

Step 4: Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:  P G KUSHALI
RegisterNumber:  212223230110
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con = confusion_matrix(y_test,y_pred)
con
cl=classification_report(y_test,y_pred)

```
## Output:

## HEAD:

![image](https://github.com/user-attachments/assets/8797ac2b-54e8-443b-8281-1c18bd8c2b18)

## INFO:

![image](https://github.com/user-attachments/assets/12c7fb64-8199-4271-928c-a0951e5e8fad)

## IS NULL:

![image](https://github.com/user-attachments/assets/da1b5a92-7172-444e-a399-ea35de847f3b)

## ACCURACY:

![image](https://github.com/user-attachments/assets/ba7a0e97-b48b-4af8-82f8-3e2c0ddb22a1)

## CONFUSION MATRIX AND CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/c3216e34-168c-4327-9072-d01b76288600)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
