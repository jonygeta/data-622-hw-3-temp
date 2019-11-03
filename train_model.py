# -*- coding: utf-8 -*-
"""
CUNY MSDS Program, DATA 622, Homework 3
Created: November 2019

@author: Yohannes Deboch 

This module validates data set, builds model trying to predict the outcome of 
Titanic disaster for each passenger based on corresponding data features.
Model is saved as pkl file. Confusion matrix, classification report and 
ROC is saved. 
"""

# Custom module for loading Titanic data set
import pull_data as loader
# Standard required packages
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stdout

def clean_data_set(df):
    """Clean and prep data set for modeling."""
    
    # Drop unnecessary columns
    df.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
    
    # Convert Sex feature into dummy variable
    df = pd.get_dummies(df, drop_first=True)
    
    # Check that remaining columns are numeric
    for column in df:
        if not is_numeric_dtype(df[column]):
            raise ValueError('Non-numeric data feature')

    # Fare value of 0 is considered a missing value - set to NaN
    df.loc[df['Fare']==0,'Fare'] = np.nan
        
    return df

def impute_data_set(df):
    """Impute missing values. 
    Missing values are only allowed in Age and Fare features."""
    
    # Check that only Age and Fare contain missing values
    if df.drop(['Age','Fare'], axis=1).isnull().sum().sum()>0:
        raise ValueError('Missing value in required feature')

    # Impute missing values by replacing them with mean value
    imp = Imputer(strategy='mean', axis=0)
    df2 = imp.fit_transform(df)
    
    # Convert imputed values back to original data frame structure
    df2 = pd.DataFrame(data=df2, columns=df.columns.values)
    for column in df2:
        df2[column] = df2[column].astype(df[column].dtype.name)
    
    return df2
    
def get_hyperparameters(X, y):
    """Return best hyperparameter values for modeling."""
    
    # Set up parameter space
    c_space = np.logspace(-5, 5)
    param_grid = {'C': c_space, 'penalty': ['l1','l2']}

    # Instantiate a logistic regression classifier
    logreg = LogisticRegression()

    # Instantiate the GridSearchCV object
    logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

    # Fit it to the data
    logreg_cv.fit(X, y)

    # Return best parameters
    return logreg_cv.best_params_
    
# MAIN EXECUTION - MODELING PIPELINE
def build_model_pipeline():
    """Execute building model pipeline 
    - reading data, 
    - training, 
    - and evaluating."""
    
    # Read data
    train_df = loader.read_data_set('train')
    
    # Validate basic structure of data set
    try:
        loader.validate_data_set(train_df)
    except:
        raise
    
    # Clean data set
    try:
        train_df = clean_data_set(train_df)
    except:
        raise
    
    # Impute missing values
    # Missing values are only allowed in Age and Fare columns
    try:
        train_df = impute_data_set(train_df)
    except:
        raise
        
    # ---------------------- MODELING ---------------------
    # This is the core part of this module, so for the most part it is not 
    # embedded in functions
    
    # Remove ID column
    train_df.drop(['PassengerId'], axis=1, inplace=True)
    
    # Split into dependent and independet variables
    y = train_df['Survived'].values
    X = train_df.drop('Survived', axis=1).values
    
    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=38)
    
    # Get optimal hyperparameters and set up model
    h_params = get_hyperparameters(X_train, y_train)
    logreg = LogisticRegression(C=h_params['C'], penalty=h_params['penalty'])
    
    # Fit the classifier to the training data
    logreg.fit(X_train, y_train)
    
    # Save model
    try: 
        pickle.dump(logreg, open('model.pkl', 'wb'))
        loader.upload_file_to_s3('model.pkl',  'model')
    except:
        raise
        
    # Predict outcomes in the testing data
    y_pred = logreg.predict(X_test)
    
    # Save model parameters, confusion matrix and classification report
    try:
        with open('model_stats.txt', 'w') as text_file:
            with redirect_stdout(text_file):
                print('MODEL:')
                print(logreg)
                print('\nCONFUSION MATRIX:')
                print(confusion_matrix(y_test, y_pred))
                print('\nCLASSIFICATION REPORT:')
                print(classification_report(y_test, y_pred))
        loader.upload_file_to_s3('model_stats.txt',  'model')
    except: 
        raise
    
    # Save ROC curve
    try: 
        y_pred_prob = logreg.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig('model_roc.png')
        loader.upload_file_to_s3('model_roc.png',  'model')
    except:
        raise
    
