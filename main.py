# -*- coding: utf-8 -*-
"""
CUNY MSDS Program, DATA 622, Homework 3
Created: October 2018

@author: Ilya Kats

This module provides main starting point for the project. 
"""

import train_model as train
import score_model as test

print('Building model...')
train.build_model_pipeline()

print('Scoring test data...')
test.score_test_pipeline()

