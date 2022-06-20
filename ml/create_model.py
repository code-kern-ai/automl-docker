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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
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
                print(">> File found! Loading file...")
                break
            else:
                print(">> File not found, please try again!")
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
                print(">> Loading columns...")
                break
            else:
                print(">> Columns not found, please try again.")
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
                print(">> Loading column")
                break
            else:
                print(">> Column not found, please try again.")  
                pass  
        except:
            pass
    return target_input
    
# get datetime dd/mm/YY H:M
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M")

# Read in the data with pandas, then convert text corpus to list
print(">> Please enter the path to where your data is stored!")
print(">> On Windows the path might look like this  ->  C:\\Users\\yourname\\data\\training_data.csv")
print(">> On MacOS/ Linux the path might look like this  ->  /home/user/data/training_data.csv")
print(" ")

# Get the path where the data is stored
PATH = file_getter()
df = pd.read_csv(PATH)
df = df.fillna('Nicht verfuegbar')

# Get the name of the features
print(" ")
print(">> Please provide the feature columns for the training data!")
print(">> You may write: column1, column 2, column_3,")
print(f">> Found columns: {df.columns}")
print(" ")
feature_columns = feature_getter(df)

# Load the data with the provided info, preprocess the text corpus
# If multiple columns are provided, the will be combinded for preprocessing
corpus = df[feature_columns]
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
    print(" ")
    print(">> Please input a number to choose your method of preprocessing the text data.")
    print(">> 1 - distilbert-base-uncased -> Very accurate, state of the art method, but slow (especially on large datasets). [ENG]")
    print(">> 2 - all-MiniLM-L6-v2 -> Faster, but still relatively accurate. [ENG]")
    print(">> 3 - Custom model -> Input your own model from https://huggingface.co/.")
    print(" ")

    choice = input("> ")
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
features = embeddings

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Param grid for random search
params = {
        'n_estimators' : [250, 300, 350, 400, 450, 500],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.01, 0.1, 0.5, 1, 1.5],
        'subsample': [0.6, 0.8, 1.0],
        'learning_rate': [0.01, 0.1],
        'max_depth': [2, 3, 4, 5, 6],
        'random_state': [42]
        }

# Instantiate and test the model
print(" ")
model = xgb.XGBClassifier()
rs_model = RandomizedSearchCV(model, param_distributions=params, n_iter=3, scoring='accuracy', cv=3, verbose=3)
rs_model.fit(X_train, y_train)
y_pred = rs_model.predict(X_test)

# Save model
with open('ml/model.pkl', 'wb') as handle:
    pickle.dump(rs_model, handle)

# Save encoder
with open('ml/encoder.pkl', 'wb') as handle:
    pickle.dump(encoder, handle)

# Generate evaluation metrics
print(" ")
print(">> Generating evaluation metrics on unseen testing data...")
print(" ")
print("- - - - - - - - - - - - - - - -")
print(f">> Model accuracy is: {round(accuracy_score(y_test, y_pred), 2) * 100} %")
print("- - - - - - - - - - - - - - - -")
print(f">> Mean squared error is: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}")
print("- - - - - - - - - - - - - - - -")
print(f">> The confusion matrix is: {confusion_matrix(y_test, y_pred)}")

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

with open('ml/config.ini', 'w') as f:
    config.write(f)
