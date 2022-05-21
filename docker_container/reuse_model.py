# Filter all debugging messages from tensorflow 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

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

# Tensorflow
import tensorflow
from tensorflow.keras.models import load_model

# load the model from a pickle (.pkl) file 
# with open('/home/leopuettmann/repos/automl-docker/Test data and pretrained model/XGBoost 16-05-2022 13-15.pkl', 'rb') as fid:
#     loaded_model = pickle.load(fid)

def load_transformer():
    """ 
    Loads in a pickled transformer model. Note that the transformer must be created
    using the 'create_model.py' file. 
    """

    transformer = pickle.load(open('transformer.pk', 'rb' ) )
    return transformer

def create_embeddings(transformer, corpus):
    """ 
    Uses loaded transformer to create word embeddings.

    Args:
    - transformer -> Loaded transformer model from the 'load transformer' function.
    - corpus -> Text corpus 
    """
    
    embeddings = transformer.transform(corpus)
    return embeddings

def create_predictions(embeddings):
    """ 
    Creates predictions with either an already trained neural net or a 
    logistic regression. 

    Args: 
    - embeddings -> Word embeddings created by the 'create_embeddings' function.
    """

    try:
        model = load_model('Neural Net')
        preds = model.predict(embeddings)
        return preds

    except:
        model =  pickle.load(open('Logistic Regression.pkl', 'rb' ) )
        preds = model.predict(embeddings)
        return preds

