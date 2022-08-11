import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

loans = pd.read_csv("classified-data.csv")

loans.drop(['Customer ID','Name','Gender','Has Active Credit Card'], axis=1,inplace=True)

sns.countplot(x='Type of Employment' , data=loans)

loans = loans.fillna(method='ffill')

final_data = pd.get_dummies(loans,columns=['Income Stability','Profession','Type of Employment','Location','Expense Type 1','Expense Type 2','Property Location'])

#print(final_data.head(0))

x = final_data.drop('Loan Sanction Amount (USD)',axis=1)
y = final_data['Loan Sanction Amount (USD)']


X_train,X_test,Y_train,Y_test= train_test_split(x,y, test_size=0.20,random_state=100)

lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y_train)

dtree = DecisionTreeClassifier()

print(dtree.fit(X_train,training_scores_encoded ))

plt.show()