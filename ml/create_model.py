# Logistic Regression spits out warnings on large datasets. For now, there warning will be surpressed.
# This warning will get taken care of in a later version
import warnings

warnings.filterwarnings("ignore")

# Basis libraries
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path

# Sklearn libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)

# Import torch libraries
import xgboost as xgb

# Embedders and Transformers
from embedders.classification.contextual import TransformerSentenceEmbedder

# Hyperparameter optimization
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(
    f"""{bcolors.BOLD}{bcolors.OKBLUE}
            __             __       __         __          
 ___ ___ __/ /____  __ _  / /______/ /__  ____/ /_____ ____
/ _ `/ // / __/ _ \/  ' \/ /___/ _  / _ \/ __/  '_/ -_) __/
\_,_/\_,_/\__/\___/_/_/_/_/    \_,_/\___/\__/_/\_\\__/_/                                                
{bcolors.ENDC}
maintained by {bcolors.BOLD}{bcolors.OKBLUE}Kern AI{bcolors.ENDC} (please visit {bcolors.UNDERLINE}https://github.com/code-kern-ai/automl-docker{bcolors.ENDC} for more information.)
"""
)

def file_getter():
    """
    Function to grab and validate an input from a user.
    Input should be a filepath to some data.

    Args:
    path_statement -> Path of the data that should be loaded.
    """
    while True:
        try:
            path_input = input("> ")
            my_file = Path(path_input)
            if my_file.is_file():
                print(f"{bcolors.OKGREEN}File found! Loading file...{bcolors.ENDC}")
                break
            else:
                print(f"{bcolors.FAIL}File not found, please try again!{bcolors.ENDC}")
                pass
        except:
            pass
    return path_input

def feature_getter(df):
    """
    Function to grab and validate multiple column names from user.
    Input should be a pandas DataFrame.

    Args:
    df -> An already loaded pandas DataFrame.
    """
    while True:
        try:
            features_input = input("> ")
            features_cleaned = [i.strip() for i in features_input.split(sep=",")]
            if pd.Series(features_cleaned).isin(df.columns).all():
                print(f"{bcolors.OKGREEN}Loading column(s)...{bcolors.ENDC}")
                break
            else:
                print(f"{bcolors.FAIL}Column(s) not found, please try again.{bcolors.ENDC}")
                pass
        except:
            pass
    return features_cleaned

def target_getter(df):
    """
    Function to grab and validate a single column name from user.
    Input should be a pandas DataFrame.

    Args:
    df -> An already loaded pandas DataFrame.
    """
    while True:
        try:
            target_input = input("> ")
            if target_input in df:
                print(f"{bcolors.OKGREEN}Loading column...{bcolors.ENDC}")
                break
            else:
                print(f"{bcolors.FAIL}Column not found, please try again.{bcolors.ENDC}")
                pass  
        except:
            pass
    return target_input
    
# get datetime dd/mm/YY H:M
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M")

# Read in the data with pandas, then convert text corpus to list
print("Please enter the path to where your data is stored!")
print(f"On {bcolors.BOLD}Windows{bcolors.ENDC} the path might look like this  ->  {bcolors.OKBLUE}C:\\Users\\yourname\\data\\training_data.csv{bcolors.ENDC}")
print(f"On {bcolors.BOLD}MacOS/Linux{bcolors.ENDC} the path might look like this  ->  {bcolors.OKBLUE}/home/user/data/training_data.csv{bcolors.ENDC}")
print()


# Get the path where the data is stored
PATH = file_getter()
df = pd.read_csv(PATH)
df = df.fillna("n/a")

# Get the name of the features
print()
print("Please provide the feature columns for the training data!")
print("You may write: column1, column 2, column_3,")
print(f"Found columns: {df.columns}")
print()
feature_columns = feature_getter(df)

# Load the data with the provided info, preprocess the text corpus
# If multiple columns are provided, the will be combinded for preprocessing
corpus = df[feature_columns]
if len(corpus.columns) > 1:
    corpus = corpus[corpus.columns].apply(
        lambda x: ",".join(x.dropna().astype(str)), axis=1
    )
    corpus = corpus.tolist()
else:
    corpus = corpus.squeeze().tolist()

# Get the names of the labels
print()
print("Please provide the column name in which the labels are store in!")
target_column = target_getter(df)
target = df[target_column].values

# Identify if the labels are strings and need encodig
if target.dtype == 'O' or str:
    encoder = LabelEncoder()
    target = encoder.fit_transform(target)
    encoder_usage = True
else:
    encoder_usage = False

# Choose a transformer model to create embeddings
while True:
    print()
    print("Please input a number to choose your method of preprocessing the text data.")
    print("1 - distilbert-base-uncased -> Very accurate, state of the art method, but slow (especially on large datasets). [ENG]")
    print("2 - all-MiniLM-L6-v2 -> Faster, but still relatively accurate. [ENG]")
    print(f"3 - Custom model -> Input your own model from {bcolors.UNDERLINE}https://huggingface.co/{bcolors.ENDC}.")
    print()

    choice = input("> ")
    if choice == '1':
        model_name = 'distilbert-base-uncased'
        print(f"Creating embeddings using {bcolors.BOLD}'{model_name}'{bcolors.ENDC} model, which will take some time. Maybe now is a good time to grab a coffee? ☕")
        sent_transformer = TransformerSentenceEmbedder(model_name)
        word_embeddings = sent_transformer.transform(corpus)
        embeddings = np.array(word_embeddings)
        break

    elif choice == "2":
        model_name = "all-MiniLM-L6-v2"
        print(
            f"Creating embeddings using {bcolors.BOLD}'{model_name}'{bcolors.ENDC}model, which will take some time. Maybe now is a good time to grab a coffee? ☕"
        )
        sent_transformer = TransformerSentenceEmbedder(model_name)
        word_embeddings = sent_transformer.transform(corpus)
        embeddings = np.array(word_embeddings)
        break

    elif choice == "3":
        print("Please input the model name you would like to use: ")
        model_name = input("> ")
        print(
            f"Creating embeddings using {bcolors.BOLD}'{model_name}'{bcolors.ENDC} model, which will take some time. Maybe now is a good time to grab a coffee? ☕"
        )
        sent_transformer = TransformerSentenceEmbedder(model_name)
        word_embeddings = sent_transformer.transform(corpus)
        embeddings = np.array(word_embeddings)
        break

    else:
        print(f"{bcolors.FAIL}Oops, that didn't work. Please try again!{bcolors.ENDC}")
        pass

# Setting up features and labels
features = embeddings

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Space
space = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    'n_estimators': hp.choice('n_estimators', [80, 90, 100, 120, 130, 140, 150, 160, 170, 200, 250, 300, 350, 400, 450, 500]),
    'max_depth' : hp.choice('max_depth', range(3, 21, 1)),
    'gamma' : hp.choice('gamma', [i/10.0 for i in range(0, 5)]),
    'colsample_bytree' : hp.choice('colsample_bytree', [i/10.0 for i in range(3, 10)]),     
    'subsample' : hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]), 
    'min_child_weight' : hp.choice('min_child_weight', [0, 0.1, 0.2, 0.5, 1, 2, 5, 10])
}

# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Objective function
def objective(params):
    xgboost = xgb.XGBClassifier(seed=0, **params)
    scores = cross_val_score(xgboost, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
    # Extract the best score
    best_score = max(scores)
    # Loss must be minimized
    loss = - best_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

# Trials to track progress
bayes_trials = Trials()

# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=6, trials=bayes_trials)
params = space_eval(space, best)

# Instantiate and test the model
print()
model = xgb.XGBClassifier(
    gamma=params['gamma'], 
    learning_rate=params['learning_rate'], 
    max_depth=params['max_depth'], 
    min_child_weight=params['min_child_weight'], 
    n_estimators=params['n_estimators'], 
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree']
)
# Fit model
model.fit(X_train, y_train)

# Predict on unseen data
y_pred = model.predict(X_test)

# Save model
with open("ml/model.pkl", "wb") as handle:
    pickle.dump(model, handle)

# Save encoder
with open("ml/encoder.pkl", "wb") as handle:
    pickle.dump(encoder, handle)

# Generate evaluation metrics
print()
print("Generating evaluation metrics on unseen testing data...")
print("- - - - - - - - - - - - - - - -")
print(f"Model accuracy is: {round(accuracy_score(y_test, y_pred), 2) * 100} %")
print("- - - - - - - - - - - - - - - -")
print(f"Mean squared error is: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}")
print("- - - - - - - - - - - - - - - -")
print(f"The confusion matrix is:")
for row in confusion_matrix(y_test, y_pred):
    print(row)

config = ConfigParser()
config['Data'] = {
    'path_to_data': PATH,
    'features': feature_columns,
    'targets': target_column,
}
config['Transformer_Model'] = {
    'model_used' : model_name
} 
config['ML_Model'] = {
    'type_ml_model' : type(model)
}
config['Encoder'] = {
    'usage' : encoder_usage
}
config["Transformer_Model"] = {"model_used": model_name}
config["ML_Model"] = {"type_ml_model": type(model)}
config["Encoder"] = {"usage": encoder_usage}

with open("ml/config.ini", "w") as f:
    config.write(f)

print(f"{bcolors.BOLD}That's it!{bcolors.ENDC} You have built your model, and can now run it with docker 🐳")
