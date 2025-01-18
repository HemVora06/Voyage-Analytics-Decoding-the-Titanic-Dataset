import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Titanic Dataset

titanic_dataset = pd.read_csv("C:\Hem\Titanic_Machine_Learning_Project\Titanic_dataset.csv")

#Inspecting Dataset

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

#Histogram for Fare
sns.histplot(titanic_dataset['Fare'], bins=30, kde=True)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')

#Examining the Heatmap of the Correlation Matrix
corr_matrix = titanic_dataset.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

#Inspecting any Non-Standard Value
print(titanic_dataset.loc[titanic_dataset['Age']==0, 'Age'])
print(titanic_dataset.loc[titanic_dataset['Age']>90, 'Age'])

#Distribution of Survivours and Non-Survivours
sns.countplot(x='Survived', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours', fontsize=18)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=16)
plt.ylabel('Count', fontsize=16)

#Distribution of Survivours and Non-Survivours based on Demographics
#By Sex
sns.countplot(x='Survived', hue='Sex', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours based on Sex', fontsize=14)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)

#By Class
sns.countplot(x='Survived', hue='Pclass', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours based on Class', fontsize=14)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)

#By Embarkment
sns.countplot(x='Survived', hue='Embarked', data=titanic_dataset, palette='pastel')
plt.title('Distribution of Survivours and Non-Survivours based on Embarkment', fontsize=14)
plt.xlabel('Survived(0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)

#Calculating mean, median, mode and standard deviation of only some specific columns
summary_stats= pd.DataFrame({
    'Mean': [titanic_dataset['Age'].mean(), titanic_dataset['Fare'].mean()]
    , 'Median': [titanic_dataset['Age'].median(), titanic_dataset['Fare'].median()]
    , 'Mode': [titanic_dataset['Age'].mode()[0], titanic_dataset['Fare'].mode()[0]]
    , 'Standard Deviation': [titanic_dataset['Age'].std(), titanic_dataset['Fare'].std()]},
    index=['Age', 'Fare'])
print(summary_stats)

#Understanding the distribution of Fares using the boxplot
sns.boxplot(x='Pclass', y='Fare', data=titanic_dataset)
'''As seen in the box plot, the First class is pretty equally distributed,
the Second class is quite Positively Skewed since the median line is quite close to the bottom,
the Third class, however, is Extremely Positively Skewed 
as the median line disappears while leaving a lot of outliers.
This is a problem for many Machine Learning Algorithms like Linear Regression that assume normality in the data,
so this has to be dealt with at the time of implementing the algorithms.'''

#Using Violin Plot to Visualize the Relationship between Age and class
sns.violinplot(x='Pclass', y='Age', data=titanic_dataset, split=True)

#Calculating the Mean Survival Rate by Class
pclass_survival_rate = titanic_dataset.groupby('Pclass')['Survived'].mean() * 100
print(pclass_survival_rate)

#Calculating Average Age and Fare as per the Class
pclass_avg_age_fare = titanic_dataset.groupby('Pclass')[['Age', 'Fare']].mean()
print(pclass_avg_age_fare)

#Investigating the Relation between Age and Fare using Scatter Plot
sns.scatterplot(x='Age', y='Fare', data=titanic_dataset)

#Calculating Mode and Median for the Embarked column
embarked_mode = titanic_dataset['Embarked'].mode()[0]
numeric_embarked=titanic_dataset['Embarked'].map({'S':1, 'C':2, 'Q':3})
embarked_median = numeric_embarked.median()
print('Embarked Mode & Median: ', embarked_mode, ', ', embarked_median)

#Summarizng the “SibSp” and “Parch” columns with descriptive statistics.
summary_stats=pd.DataFrame({
    'Minimum':[titanic_dataset['SibSp'].min(), titanic_dataset['Parch'].min()],
    'Maximum':[titanic_dataset['SibSp'].max(), titanic_dataset['Parch'].max()],
    'Mean':[titanic_dataset['SibSp'].mean(), titanic_dataset['Parch'].mean()],
    'Median':[titanic_dataset['SibSp'].median(), titanic_dataset['Parch'].median()],
    'Mode':[titanic_dataset['SibSp'].mode()[0], titanic_dataset['Parch'].mode()[0]]
}, 
index=['Sibsp', 'Parch'])
print(summary_stats)

#Creating a new feature 'Family Size'
titanic_dataset['Family Size'] = titanic_dataset['SibSp'] + titanic_dataset['Parch'] + 1
print(titanic_dataset['Family Size'])

#Creating a pivot table to summarize survival rates across different combinations of "Pclass" and "Sex"
pivot_table = pd.pivot_table(data=titanic_dataset, index='Pclass', values='Survived', columns='Sex', aggfunc='mean', margins=True)
print(pivot_table)

#Plotting a bar chart to show the survival rate by “Sex” and “Pclass”
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_dataset)
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')

#Exploring the relationships between numerical and categorical variables using box plots
sns.boxplot(x='Embarked', y='Fare', data=titanic_dataset)

#Visualize the distribution of “Age” by “Survived” using a box plot
sns.boxplot(x='Survived', y='Age', data=titanic_dataset)

#Explore the relationship between “Age” and “Fare” through a correlation matrix
corr_matrix_age_fare=titanic_dataset[['Age', 'Fare']].corr(numeric_only=True)
print(corr_matrix_age_fare)

#Group by “Pclass” and plot the distribution of “Fare” for each class
sns.boxplot(x='Pclass', y='Fare', data=titanic_dataset)

#19. Calculate and visualize the proportion of each age group (“Child”, “Adult”, “Senior”).
#   - Child: 0-19
#   - Adult: 20-64
#   - Senior: 65-120
titanic_dataset['Age Group'] = pd.cut(titanic_dataset['Age'], bins=[0, 20, 65, 120], labels=['Child', 'Adult','Senior'], include_lowest=True)
print(titanic_dataset['Age Group'])

#Create a visualization showing the relationship between family size and survival rate
sns.barplot(x='Family Size', y='Survived', data=titanic_dataset)

#Normalize and standardize the "Fare" column
class MinMaxScaler:
    def __init__(self, feature_range):
        self.feature_range = feature_range
        self.min_=None
        self.max_=None

    def fit(self, X):
        #Find Min and Max of the Feature
        self.min_=X.min(axis=0)
        self.max_=X.max(axis=0)
        return self

    def transform(self, X):
        #Apply MinMax formula
        scaled=((X-self.min_)/(self.max_-self.min_))
        scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled

    def fit_transform(self, X):
        #Add Fit and Transform Functions
        self.fit(X)
        return self.transform(X)

scaler=MinMaxScaler(feature_range=(0, 1))
titanic_dataset['Normalized_Fare']=scaler.fit_transform(titanic_dataset['Fare'])
print(titanic_dataset['Normalized_Fare'].head())

class StandardScaler:
    def __init__(self):
        self.mean_=None
        self.standard_deviation_=None
    def fit(self, X):
        #Find Mean and Standard Deviation of the Feature
        self.mean_=X.mean()
        self.standard_deviation_=X.std()
        return self
    def transform(self, X):
        #Apply Standardization formula
        scaled=(X-self.mean_)/self.standard_deviation_
        return scaled
    def fit_transform(self, X):
        #Add Fit and Transform Functions
        self.fit(X)
        return self.transform(X)

scaler2=StandardScaler()
titanic_dataset['Standardized_Fare']=scaler2.fit_transform(titanic_dataset[['Fare']])
print(titanic_dataset['Standardized_Fare'].head())

#One hot encoding the sex and embarked columns
titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Sex', 'Embarked'])
titanic_dataset['Embarked_C'] = titanic_dataset['Embarked_C'].astype('int')
titanic_dataset['Embarked_S'] = titanic_dataset['Embarked_S'].astype('int')
titanic_dataset['Embarked_Q'] = titanic_dataset['Embarked_Q'].astype('int')
titanic_dataset['Sex_male']=titanic_dataset['Sex_male'].astype('int')
titanic_dataset['Sex_female']=titanic_dataset['Sex_female'].astype('int')
print(titanic_dataset.head())

#Removing columns that are not useful for analysis (e.g., "Name", "Ticket")
titanic_dataset.drop('Name', axis=1, inplace = True)
titanic_dataset.drop('Ticket', axis=1, inplace = True)
titanic_dataset.drop('Cabin', axis=1, inplace = True)

#Scaling the numeric features "Age" MinMaxScaler and StandardScaler
titanic_dataset['Normalized_Age']=scaler.fit_transform(titanic_dataset['Age'])
print(titanic_dataset['Normalized_Age'].head())
titanic_dataset['Standardized_Age']=scaler2.fit_transform(titanic_dataset['Age'])
print(titanic_dataset['Standardized_Age'].head())

#Perform a final inspection of the dataset after cleaning to ensure accuracy and consistency
print(titanic_dataset.head())
print(titanic_dataset.describe(include=[np.number]))

#Creating new 'IsAlone' feature
titanic_dataset['IsAlone'] = (titanic_dataset['SibSp'] == 0) & (titanic_dataset['Parch'] == 0)
titanic_dataset['IsAlone'] = titanic_dataset['IsAlone'].astype('int')
print(titanic_dataset['IsAlone'])

#Creating a Recursive Feature Eliminator for optimum Feature Selection from Scratch

    #Creating a Machine Learning model to search for the optimum Features
     