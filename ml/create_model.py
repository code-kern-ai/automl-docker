# Logistic Regression spits out warnings on large datasets. For now, there warning will be surpressed.
# This warning will get taken care of in a later version
import warnings
warnings.filterwarnings("ignore")

# Basis libraries
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from configparser import ConfigParser

# Sklearn libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)

# Import torch libraries
import xgboost as xgb

# Embedders and Transformers
from embedders.classification.contextual import TransformerSentenceEmbedder

# def my_accuracy_scorer(*args):
#     score = accuracy_score(*args)
#     print('Training ... current score is {}'.format(round(score, 2)))
#     return score

# custom_score = make_scorer(my_accuracy_scorer, greater_is_better=True)

print(" ")
print("  ___________      _  __                      ")
print(" |  _______  |    | |/ /                 /\  |")
print(" | |       | |    |   / ___ _ __ _ __   /‾‾\ |")
print("| |        | |    |  < / _ \ |__| |_ \        ")
print("| |_______| |     | . \  __/ |  | | | |       ")
print("|___________|     |_|\_\___|_|  |_| |_|       ")
print(" ")
print("AutoML Tool by kern.ai ©")
print("Please visit https://github.com/code-kern-ai/automl-docker for instructions.")
print(" ")


def input_getter(path_statement):
    """
    Function to grab and validate an input from a user.
    Input should be a filepath to some data.

    Args:
    path_statement -> Path of the data that should be loaded.
    """
    while True:
        user_input = input()
        print(path_statement, user_input)
        path_approval = input("(y/ n) ")
        if path_approval.lower() == "y":
            break
        elif path_approval.lower() == "n":
            print(">> Enter a new path: ")
            print(" ")
        else:
            print(">> Sorry, that didn't work. Please enter again: ")
            print(" ")
    return user_input


# get datetime dd/mm/YY H:M
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M")

# Read in the data with pandas, then convert text corpus to list
print(">> Please select the path to where your data is stored!")
print(" ")
print(">> On Windows the path might look like this  ->  C:\\Users\\yourname\\data\\training_data.csv")
print(">> On MacOS/ Linux the path might look like this  ->  home/user/data/training_data.csv")
print(" ")

# Get the path where the data is stored
PATH = input_getter(">> Is this the correct path? ->")
df = pd.read_csv(PATH)
df = df.fillna('Nicht verfuegbar')
print(">> Data successfully loaded!")


# Get the name of the features
print(" ")
print(">> Please provide one or multiple column names!")
COL_TEXTS = input_getter(">> Is this the correct column name? ->")

# Load the data with the provided info, preprocess the text corpus
# If multiple columns are provided, the will be combinded for preprocessing
corpus = df[COL_TEXTS.split()]
if len(corpus.columns) > 1:
    corpus = corpus[corpus.columns].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
    )
    corpus = corpus.tolist()

else:
    corpus = corpus.squeeze().tolist()

# Get the names of the labels
print(" ")
print(">> Please provide the column name in which the labels are store in!")
COL_LABEL = input_getter(">> Is this the correct column name? ->")

# Choose a transformer model to create embeddings
while True:
    print(" ")
    print(">> Please input a number to choose your method of preprocessing the text data.")
    print(">> 1 - distilbert-base-uncased -> Very accurate, state of the art method, but slow (especially on large datasets). [ENG]")
    print(">> 2 - all-MiniLM-L6-v2 -> Faster, but still relatively accurate. [ENG]")
    print(">> 3 - Custom model -> Input your own model from https://huggingface.co/.")
    print(" ")

    choice = input()
    if choice == '1':
        model_name = 'distilbert-base-uncased'
        print(f">> Creating embeddings using '{model_name}' model, this might take a couple of minutes ...")
        sent_transformer = TransformerSentenceEmbedder(model_name)
        word_embeddings = sent_transformer.transform(corpus)
        embeddings = np.array(word_embeddings)
        break

    elif choice == '2':
        model_name = 'all-MiniLM-L6-v2'
        print(f">> Creating embeddings using '{model_name}' model, this might take a couple of minutes ...")
        sent_transformer = TransformerSentenceEmbedder(model_name)
        word_embeddings = sent_transformer.transform(corpus)
        embeddings = np.array(word_embeddings)
        break

    elif choice == '3':
        model_name = input('>> Please input the model name you would like to use: ')
        print(f">> Creating embeddings using {model_name} model, this might take a couple of minutes ...")
        sent_transformer = TransformerSentenceEmbedder(model_name)
        word_embeddings = sent_transformer.transform(corpus)
        embeddings = np.array(word_embeddings)
        break

    else:
        print(">> Oops, that didn't work. Please try again!")
        pass

# Setting up features and labels
target = df[COL_LABEL].values
features = embeddings

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Param grid for random search
params = {
        'n_estimators' : [200, 250, 300, 400, 450, 500, 550, 600, 650, 700],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.01, 0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'learning_rate': [0.001, 0.005, 0.01, 0.1],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'random_state': [42]
        }

# Instantiate and test the model
model = xgb.XGBClassifier()
rs_model = RandomizedSearchCV(model, param_distributions=params, n_iter=10, scoring='roc_auc', n_jobs=4, cv=3, verbose=3, random_state=42)
rs_model.fit(X_train, y_train)
y_pred = rs_model.predict(X_test)

# Save model
with open('ml/model.pkl', 'wb') as handle:
    pickle.dump(rs_model, handle)

# Generate evaluation metrics
print(" ")
print(">> Generating evaluation metrics on unseen testing data...")
print(" ")
print("- - - - - - - - - - - - - - - -")
print(f">> Model accuracy is: {round(accuracy_score(y_test, y_pred), 2) * 100} %")
print("- - - - - - - - - - - - - - - -")
print(f">> Mean squared error is: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}")
print("- - - - - - - - - - - - - - - -")
print(f">> AUC is: {round(roc_auc_score(y_test, y_pred), 2)}")
print("- - - - - - - - - - - - - - - -")
print(f">> The confusion matrix is: {confusion_matrix(y_test, y_pred)}")

config = ConfigParser()
config['Data'] = {
    'path_to_data': PATH,
    'features': COL_TEXTS,
    'targets': COL_LABEL,
}
config['Transformer_Model'] = {
    'model_used' : model_name
} 
config['ML_Model'] = {
    'type_ml_model' : type(model)
}

with open('ml/config.ini', 'w') as f:
    config.write(f)
