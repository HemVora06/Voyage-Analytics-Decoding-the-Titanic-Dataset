'''This is for practicing new algorithms with smaller and much simpler codebases'''
import pandas as pd
import numpy as np

main_titanic_dataset=pd.read_csv('Cleaned_Titanic_Dataset.csv')
titanic_datset=main_titanic_dataset.loc[:50, :]
features=[titanic_datset.Pclass, titanic_datset.Age, titanic_datset.Sex, titanic_datset.Fare]
#Makin a Decision Tree Classifier from Scratch
class DecisionTreeClassifier:
    def __init__(self, max_depth, min_split):
        self.max_depth = max_depth
        self.min_split = min_split
        self._left_data=None
        self._right_data=None
        self._gini_total=None
        self._left_impurity=None
        self._right_impurity=None


    def split_data(self, dataset, features, threshold):
        #Splitting the data
        left_data = dataset[dataset[features] <= threshold][['Age', 'Survived']]
        right_data = dataset[dataset[features] > threshold][['Age', 'Survived']]
        self._left_data=left_data
        self._right_data=right_data
        return self._left_data, self._right_data


    def calculate_impurity(self, dataset, features, threshold):
        #Calculating Impurities in a node
        left_data, right_data = self.split_data(dataset, features, threshold)

        #Gini Impurity Calculation Function
        def gini_impurity(dataset):
            #No impurity if no data
            if len(dataset) == 0:
                return 0 
            #Calculating Class Probabilities
            class_probabilities=dataset['Survived'].value_counts(normalize=True).values
            #Applying Gini Impurity Equation
            return 1 - sum((p**2) for p in class_probabilities)
        
        #Calculating Gini Impurity for left data
        left_impurity=gini_impurity(left_data)
        #Calculating Gini Impurity of right data
        right_impurity=gini_impurity(right_data)
        #Calculating the weighted average of the impurities of the left and right child nodes
        gini_total=((len(left_data)/(len(left_data)+len(right_data)))*left_impurity)+((len(right_data)/(len(left_data)+len(right_data)))*right_impurity)
        self._left_impurity=left_impurity
        self._right_impurity=right_impurity
        self._gini_total=gini_total
        return gini_total
    

    def get_left_impurity(self):
        return self._left_impurity
    

    def get_right_impurity(self):
        return self._right_impurity


    def get_gini_total(self):
        return self._gini_total


    def get_left_data(self):
        return self._left_data
    
    
    def get_right_data(self):
        return self._right_data
    
DecisionTree=DecisionTreeClassifier(max_depth=2, min_split=3)
left_data, right_data=DecisionTree.split_data(titanic_datset, 'Age', 30)
weighted_gini=DecisionTree.calculate_impurity(titanic_datset, 'Age', 30)
print(DecisionTree.get_left_data())
print(DecisionTree.get_right_data())
print(DecisionTree.get_gini_total())
print(DecisionTree.get_left_impurity())
print(DecisionTree.get_right_impurity())

    