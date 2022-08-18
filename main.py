import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model

def imputeIncomeSta(cols):
    InSt = cols[0]
    pro = cols[1]
    if pd.isnull(InSt):
        if pro == "Pensioner":
            return("High")
        else:
            return("Low")
    else:
        return InSt


def computeTOEHF(cols):
    pro = cols['Profession']
    TOE = cols['Type of Employment']

    if pd.isnull(TOE):
        if pro == "Pensioner":
            return("Retired")
    else:
        return TOE
loans=pd.read_csv('classified-data.csv')
loans.drop(['Customer ID','Name','Gender','Property ID'], axis=1,inplace=True)

loans = loans.dropna(subset = ['Loan Sanction Amount (USD)'])

#sns.heatmap(loans.corr(),square=True)

#sns.scatterplot(y=loans['Property Price'], x=loans['Income (USD)'])

#sns.scatterplot(y=loans['Loan Sanction Amount (USD)'], x=loans['Property Price'])

#sns.scatterplot(y=loans['Loan Sanction Amount (USD)'], x=loans['Loan Amount Request (USD)'])

#sns.scatterplot(y=loans['Loan Sanction Amount (USD)'], x=loans['Current Loan Expenses (USD)'])

#sns.scatterplot(y=loans['Loan Sanction Amount (USD)'], x=loans['Credit Score'])


loans['Income Stability'] = loans[['Income Stability','Profession']].apply(imputeIncomeSta,axis=1)
loans['Property Age'] = loans['Property Age'].fillna(loans['Income (USD)'])

loans['Type of Employment'] = loans[['Profession','Type of Employment']].apply(computeTOEHF , axis =1)
loans['Current Loan Expenses (USD)'].fillna(value=loans['Current Loan Expenses (USD)'].mean(),inplace=True)

loans['Credit Score'].fillna(value=loans['Credit Score'].mean(),inplace=True)


loans['Property Age'].fillna(value=loans['Income (USD)'].mean(),inplace=True)


loans['Property Location'].fillna(loans['Property Location'].mode()[0], inplace=True)
loans['Has Active Credit Card'].fillna(loans['Has Active Credit Card'].mode()[0], inplace=True)
loans['Income (USD)'].fillna(value=loans['Income (USD)'].mean(),inplace=True)
loans['Dependents'].fillna(value=2,inplace=True)





final_data = pd.get_dummies(loans,columns=['Income Stability','Profession','Type of Employment','Location','Expense Type 1','Expense Type 2','Has Active Credit Card','Property Location'])


x = final_data.drop('Loan Sanction Amount (USD)',axis=1)
y = final_data['Loan Sanction Amount (USD)']

X_train,X_test,Y_train,Y_test= train_test_split(x,y, test_size=0.20,random_state=100)

#Linear regressor
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

ly_pred= regressor.predict(X_test)
print("Linear Regression ",mean_squared_error(Y_test,ly_pred))
print("r2_score Linear Regression :",r2_score(Y_test,ly_pred))


#Decision Tree
dtree = DecisionTreeRegressor()

dtree.fit(X_train,Y_train )
pred_y = dtree.predict(X_test)

print("Decision Tree Regression ",mean_squared_error(Y_test,pred_y))
print("r2_score Decision Tree Regression:",r2_score(Y_test,pred_y))

#random forest
rfc = RandomForestRegressor()
rfc.fit(X_train,Y_train)
pred_y1 = rfc.predict(X_test)

print("Random forest Regression ",mean_squared_error(Y_test,pred_y1))
print("r2_score Random forest Regression :",r2_score(Y_test,pred_y1))

#support vector

model = SVR()
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
print("Support vector Regression ",mean_squared_error(Y_test,predictions))
print("r2_score Support vector Regression :",r2_score(Y_test,predictions))

#Bayesian Regression Model

reg = linear_model.BayesianRidge()
reg.fit(X_train, Y_train)
predic = reg.predict(X_test)
print("Bayesian Regression Model ",mean_squared_error(Y_test,predic))
print("r2_score Bayesian Regression Model:",r2_score(Y_test , predic))
