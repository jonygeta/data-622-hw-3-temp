# -*- coding: utf-8 -*-
"""
CUNY MSDS Program, DATA 622, Homework 3
Created: October 2018

@author: Ilya Kats

This module loads model from pkl file, imports test data and runs prediction.
"""

# Custom module for loading Titanic data set
import pull_data as loader
from train_model import clean_data_set, impute_data_set
# Standard required packages
import pickle
import pandas as pd

def score_test_pipeline():
    """Run scoring pipeline 
    - read test data,
    - load model,
    - evaluate and write results."""
    
    # Read data
    test_df = loader.read_data_set('test')
    
    # Validate basic structure of data set
    try:
        loader.validate_data_set(test_df, target_exists=False)
    except:
        raise
    
    # Clean data set
    try:
        test_df = clean_data_set(test_df)
    except:
        raise
    
    # Impute missing values
    # Missing values are only allowed in Age and Fare columns
    try:
        test_df = impute_data_set(test_df)
    except:
        raise
    
    # Load model from the pickle file
    try:
        loader.download_file_from_s3('model.pkl', 'model')
        logreg = pickle.load(open('model.pkl', 'rb'))
    except:
        raise
    
    # Remove passenger ID
    p_df = test_df.drop(['PassengerId'], axis=1)
    
    # Run prediction
    pred_df = logreg.predict(p_df)
    
    # Concatenate passenger ID and prediction
    pred_df = pd.concat([test_df['PassengerId'],pd.DataFrame(pred_df)], axis=1)
    pred_df.columns = ['PassengerId', 'Survived']
    
    # Save results to CSV file for Kaggle submission
    try:
        pred_df.to_csv('kaggle_submission.csv', index=False)
        loader.upload_file_to_s3('kaggle_submission.csv')
    except:
        raise