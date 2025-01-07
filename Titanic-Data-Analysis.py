import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

'''Importing Titanic Dataset'''

titanic_dataset = pd.read_csv("C:\Hem\Titanic_Machine_Learning_Project\Titanic_dataset.csv")

'''Inspecting Dataset'''

print(titanic_dataset.head())
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

'''Basic Exploratary Analysis:'''

#Checking For Duplicates
print(titanic_dataset.duplicated().sum()) #0

# Identifying anymore inconsistencies
print(titanic_dataset.describe()) #Summary of the dataset

#Calculating the Z Score of Fare
titanic_dataset['fare_zscore'] = stats.zscore(titanic_dataset['Fare'])

#Checking for outliers
print(titanic_dataset.loc[titanic_dataset['Fare']==0, 'Age'])

#Since, fare for someone older than a todller can't be 0, we'll cange it by implementing Fare Based Dynamic Thresholding
group_mean=titanic_dataset.groupby('Age')['Fare'].transform('mean')
titanic_dataset['Fare']=titanic_dataset.apply(lambda x:group_mean[x.name] if x['Fare']==0 else x['Fare'], axis=1)

#Checking if the outliers are reduced:
print(titanic_dataset.loc[titanic_dataset['Fare']==0, 'Age'])
#Outliers Reduced

#Checking for outliers
print('Outliers: \n', titanic_dataset[titanic_dataset['fare_zscore'] >= 3][['Fare', 'fare_zscore']])


#Implementing Z-Score Based Dynamic Thresholding
group_mean=titanic_dataset.groupby('Pclass')['Fare'].transform('mean')
titanic_dataset['Fare']=titanic_dataset.apply(lambda x:group_mean[x.name] if x['fare_zscore']>=3 else x['Fare'], axis=1)

#Checking if the outliers are reduced:
print(titanic_dataset[titanic_dataset['fare_zscore'] >= 3][['Fare', 'fare_zscore']])
'''The zscore doesn't fall because for it to change, z score needs to be calculated again and there will
be outliers in that as well and this will result to overfitting of the data. 
So as long as the fare has changed, it's all good.'''

#Visualizing the data

#Histogram for Age
sns.histplot(titanic_dataset['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#Histogram for Fare
sns.histplot(titanic_dataset['Fare'], bins=30, kde=True)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

#Examining the Heatmap of the Correlation Matrix
corr_matrix = titanic_dataset.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#Inspecting any Non-Standard Value
print(titanic_dataset.loc[titanic_dataset['Age']==0, 'Age'])
print(titanic_dataset.loc[titanic_dataset['Age']>90, 'Age'])

#Distribution of Survivours and Non-Survivours
sns.countplot(x='Survived', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours', fontsize=18)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.show()

#Distribution of Survivours and Non-Survivours based on Demographics
#By Sex
sns.countplot(x='Survived', hue='Sex', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours based on Sex', fontsize=14)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

#By Class
sns.countplot(x='Survived', hue='Pclass', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours based on Class', fontsize=14)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

#By Embarkment
sns.countplot(x='Survived', hue='Embarked', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours based on Embarkment', fontsize=14)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

#Calculating mean, median, mode and standard deviation of nly some specific columns
summary_stats= pd.DataFrame({
    'Mean': [titanic_dataset['Age'].mean(), titanic_dataset['Fare'].mean()]
    , 'Median': [titanic_dataset['Age'].median(), titanic_dataset['Fare'].median()]
    , 'Mode': [titanic_dataset['Age'].mode()[0], titanic_dataset['Fare'].mode()[0]]
    , 'Standard Deviation': [titanic_dataset['Age'].std(), titanic_dataset['Fare'].std()]},
    index=['Age', 'Fare'])
print(summary_stats)

#Understanding the distribution of Fares using the boxplot
sns.boxplot(x='Pclass', y='Fare', data=titanic_dataset)
plt.show()