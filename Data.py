import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("E:/Diabetes_Prediction/diabetes.csv")

print(data.shape)

print(data.head(5))

print(data.isnull().values.any())

#Giving the correlation between variables

import seaborn as sns

correlation_variable = data.corr()

#plotting the heatmap of correlated variables

#________________________________________________

#checking that data is balanced or not by counting number of 1's and 0's in diabetes

count1s = 0
count0s = 0
for outcome in data['Outcome']:
    if outcome == 1:
        count1s += 1
    else:
        count0s += 1


print(count1s)
print(count0s)

#count1s is 268 and count0s is 500 ,it has ratio of 0.5 , so we can say that it is not properly unbalancea it is quietly balanced
import seaborn as sns
top_features=correlation_variable.index
plt.figure(1,figsize=(20,20))
heatmap=sns.heatmap(data[top_features].corr(),annot=True,cmap='RdYlGn')
#Now Appliying Machine Learning Algorithm
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']
labels = ['Outcome']

print("Number of Rows missing gulcose_data:{0}".format(len(data.loc[data['Glucose'] == 0])))
print("Number of Rows missing BloodPressure_data:{0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("Number of Rows missing SkinThickness_data:{0}".format(len(data.loc[data['SkinThickness'] == 0])))
print("Number of Rows missing Insulin_data:{0}".format(len(data.loc[data['Insulin'] == 0])))
print("Number of Rows missing BMI_data:{0}".format(len(data.loc[data['BMI'] == 0])))
print("Number of Rows missing DiabetesPedigreeFunction_data:{0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("Number of Rows missing Age:{0}".format(len(data.loc[data['Age'] == 0])))

#correcting missing data
from process_missing_data import processdata

data['Insulin'] = processdata(data['Age'], data['Insulin'], 2, 3,"Age","Insulin")
data['SkinThickness']=processdata(data['Age'], data['SkinThickness'], 4, 5, "Age", "Skin Thickness")
data['BMI'] = processdata(data['Age'], data['BMI'], 6, 7,"Age", "BMI")
data['BloodPressure']=processdata(data['Age'], data['BloodPressure'], 8, 9, "Age", "BloodPressure")
data['Glucose']=processdata(data['Age'],data['Glucose'],10,11,"Age","Glucose")

#Now predicting values

features_values = data[features].values
labels_values = data[labels].values

from sklearn.model_selection import train_test_split

Features_train, Features_test, labels_train, labels_test = train_test_split(features_values, labels_values, test_size=0.3, random_state=10)

from sklearn import tree
from sklearn.metrics import accuracy_score

classifier = tree.DecisionTreeClassifier(min_samples_split=100)
classifier.fit(Features_train, labels_train)
pred = classifier.predict(Features_test)
print(accuracy_score(pred, labels_test))

plt.show()
