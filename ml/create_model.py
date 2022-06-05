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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)

# Import torch libraries
import torch
import torch.nn as nn 
import torch.optim as optim

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
print(">> Data successfully loaded!")


# Get the name of the features
print(" ")
print(">> Please provide the column name in which the texts are store in!")
COL_TEXTS = input_getter(">> Is this the correct column name? ->")

# Load the data with the provided info, lower all words in the corpus
corpus = df[COL_TEXTS].to_list()

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
y = df[COL_LABEL].values

target = torch.from_numpy(y)
target = target.unsqueeze(1)
target = target.float()

features = torch.from_numpy(embeddings)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Simple dense neural network with relu activation function
input_shape = X_train.shape[1]
model = nn.Sequential(nn.Linear(input_shape, 128),
                      nn.ReLU(),
                      nn.Linear(128, 1),
                      nn.Sigmoid())

# Set up loss funtion and optimizer to enable the model to learn
criteron = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model with 100 epochs
epochs = 100
losses = []
for epoch in range(epochs):
    pred_y = model(X_train.float())
    loss = criteron(pred_y, y_train)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()

# Create predictions on unseen data for evaluation
y_pred = model(X_test.float())
y_pred = y_pred.detach().cpu().numpy()
y_pred = (y_pred > 0.5).astype('int32')

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
