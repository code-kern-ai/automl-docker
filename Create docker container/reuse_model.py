# Basis libraries
import numpy as np 
import pandas as pd 
import pickle
import json
import os
from datetime import datetime

# Sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, mean_squared_error

# XGBoost
import xgboost as xgb

# load the model from a pickle (.pkl) file 
with open('XGBoost 16-05-2022 13-15.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

# Load the transformer or vectorizer
transformer = pickle.load(open("transformer.pk", "rb" ) )

# Load the data from json file (late provided by api)
with open('make_preds.json') as f:
    data = json.load(f)


corpus = data['text']

# Create the right embeddings
try: 
    embeddings = transformer.transform(corpus)
except:
    embeddings = transformer.encode(corpus)

preds = loaded_model.predict(embeddings)
print(preds)

