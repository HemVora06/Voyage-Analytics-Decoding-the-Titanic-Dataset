import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Importing Titanic Dataset'''

titanic_dataset = pd.read_csv("C:\Hem\Titanic_Machine_Learning_Project\Titanic_dataset.csv")

'''Inspecting Dataset'''

#print(titanic_dataset.head())
print(titanic_dataset.dtypes)

'''Finding and Handelling Missing values'''

#Finding Missing Value
print("Null Data: \n", titanic_dataset.isnull().sum())

#Solution #1:
notnull_dataset=titanic_dataset.dropna()
print("Improved Dataset: \n", notnull_dataset)
print("Null Data: \n", notnull_dataset.isnull().sum())
#Bad Solution. Loss of valuable data

#Solution #2:
titanic_dataset['Age'].fillna(titanic_dataset['Age'].median(), inplace=True) #Fills the null age value with skewed data
titanic_dataset['Embarked'].fillna(titanic_dataset['Embarked'].mode()[0], inplace=True) #Fills the null embarked value with tha most frequent occuring values
titanic_dataset['Cabin'].fillna('Unknown', inplace=True) #Places a place holder since no proper method to deal with this column
print("Null Data: \n", titanic_dataset.isnull().sum())
print("Improved Dataset: \n", titanic_dataset)
#An Acceptable Solution.

#Changing the datatype of categorical columns to 'Category' from 'Object'

titanic_dataset['Sex']=titanic_dataset['Sex'].astype('category')
titanic_dataset['Survived']=titanic_dataset['Survived'].astype('category')
titanic_dataset['Pclass']=titanic_dataset['Pclass'].astype('category')
titanic_dataset['Embarked']=titanic_dataset['Embarked'].astype('category')
print(titanic_dataset.dtypes)

#Checking for Unique values in Categorical Columns

for column in titanic_dataset.select_dtypes(include='category').columns:
    print(f"{column}: {titanic_dataset[column].unique()}")

plt.plot(titanic_dataset['Age'], titanic_dataset['Fare'])
plt.show()