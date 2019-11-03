# -*- coding: utf-8 -*-
"""
CUNY MSDS Program, DATA 622, Homework 3
Created: November 2019

@author: Yohannes Deboch 

This module reads Titanic data set from public GitHub or AWS S3.
"""

import pandas as pd
import boto3
import io

# Define data location here for ease of modification
root = 'https://raw.githubusercontent.com/ilyakats/CUNY-DATA622/master/HW2/TitanicData/'
bucket = 'data622-hw3-kats'

def read_data_set(set_name, read_from_aws=False):
    """Read data from GitHub and return data frame."""
    
    # Confirm that set name is something expected
    if set_name not in ['train','test']:
        raise ValueError('Set name must be either train or test')
    
    # Read the CSV file
    # Data can be read from GitHub or from AWS S3
    try:
        if read_from_aws:
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=bucket, Key=set_name+'.csv')
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))

        else:
            df = pd.read_csv(root+set_name+'.csv')
    except:
        print('Error downloading the data set')
        raise

    # Return data frame if successfully reached this point
    return df

def validate_data_set(df, target_exists=True):
    """Validate that data frame is in expected format and 
    some data exist.
    
    Expectations:
        - At least one row
        - 12 columns (11 if test set)
        - Columns are named as expected
    Can be expanded as needed to make it more robust."""
    
    # Confirm that there is data
    if df.shape[0]<1:
        raise ValueError('No observations found')
        
    # Confirm right number of columns
    # Only some columns are used for modeling, so this can be modified
    # to look for specific columns. However, this way is more generic - 
    # no need to update function if modeling is updated to use other features
    if (not target_exists)+df.shape[1]!=12:
        raise ValueError('Incorrect number of features')

    # Check that all columns are named as expected
    # If target is present, data frame must be identified as training set
    for column in df:
        if column == 'Survived' and not target_exists:
            raise ValueError('Target variable in test data set')            
        elif column not in ['PassengerId','Survived','Pclass','Name','Sex',
                          'Age','SibSp','Parch','Ticket','Fare',
                          'Cabin','Embarked']:
            raise ValueError('Unexpected features')

def upload_file_to_s3(file_name, folder_name=''):
    """Uploads file from local folder to AWS S3."""
    if folder_name!='':
        folder_name = folder_name+'/'
    try:
       s3 = boto3.resource('s3')
       s3.meta.client.upload_file(file_name, bucket, folder_name+file_name)
    except:
        print('Error uploading file to S3')
        raise

def download_file_from_s3(file_name, folder_name=''):
    """Downloads file to local folder from AWS S3."""
    try:
       s3 = boto3.resource('s3')
       s3.meta.client.download_file(bucket, folder_name+'/'+file_name, file_name)
    except:
        print('Error uploading file to S3')
        raise

# This module can be improved by making it less specific to this project.
# Errors are captured, but not handled - they are simply re-raised.
# With further development, errors should be logged.
