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

# Sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)

# Embedders and Transformers
from sentence_transformers import SentenceTransformer

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
print(
    ">> On Windows the path might look like this  ->  C:\\Users\\yourname\\data\\training_data.csv"
)
print(
    ">> On MacOS/ Linux the path might look like this  ->  home/user/data/training_data.csv"
)
print(" ")

# Get the path where the data is stored
PATH = input_getter(">> Is this the correct path? ->")
df = pd.read_csv(PATH)
print(">> Data successfully loaded!")

# Get the name of the features
print(">> Please provide the column name in which the texts are store in!")
print(" ")
COL_TEXTS = input_getter(">> Is this the correct column name? ->")

# Load the data with the provided info, lower all words in the corpus
corpus = df[COL_TEXTS].to_list()
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()

# Get the names of the labels
print(">> Please provide the column name in which the labels are store in!")
print(" ")
COL_LABEL = input_getter(">> Is this the correct column name? ->")


while True:
    print(" ")
    print(
        ">> Please input a number to choose your method of preprocessing the text data."
    )
    print(
        ">> 1 - Very accurate, state of the art method, but slow (especially on large datasets)"
    )
    print(">> 2 - Fast but less accurate (but still good)")
    print(" ")

    choice = input()

    if choice == "1":
        # Instantiate a sentence transformer and create embeddings
        print(
            ">> Creating embeddings using transformer model, this might take a couple of minutes ..."
        )
        sent_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeds = sent_transformer.encode(corpus, show_progress_bar=True)
        embeddings = np.array(embeds)

        # Pickle transformer to be reusable in a different file
        with open("transformer.pk", "wb") as fin:
            pickle.dump(sent_transformer, fin)
        break

    else:
        # Intantiate a tf-idf vectorizer
        print(">> Creating TF-IDF embeddings ...")
        vect = TfidfVectorizer()
        embeddings = vect.fit_transform(corpus).astype("float32")

        # Pickle vectorizer to be reusable in a different file
        with open("./docker_container/transformer.pk", "wb") as fin:
            pickle.dump(vect, fin)
        break

features = embeddings

labels = df[COL_LABEL]
labels = np.array(labels)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

print(">> Training maschine learning model ...")
print(" ")
lr = LogisticRegression(dual=False)

# Hyper parameter space is relatively small

hyperparameters = {
    "C": np.arange(0, 4),
    "penalty": ["l2", "none"],
    "max_iter": [100, 150, 250, 500],
}

# Initiate and fit random search cv
lr_clf = RandomizedSearchCV(
    estimator=lr, param_distributions=hyperparameters, cv=10, n_iter=10, verbose=2
)
lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict(X_test)

# Save the model to current directory
with open(f"./docker_container/Logistic Regression.pkl", "wb") as fid:
    pickle.dump(lr_clf, fid)
    print(" ")
    print(f">> Saved model to {os.path.abspath(os.getcwd())}")
    print(" ")

# Generate evaluation metrics
print(" ")
print(">> Generating evaluation metrics on unseen testing data...")
print(" ")
print("- - - - - - - - - - - - - - - -")
print(f">> Model accuracy is: {round(accuracy_score(y_test, y_pred), 2) * 100} %")
print("- - - - - - - - - - - - - - - -")
print(
    f">> Mean squared error is: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}"
)
print("- - - - - - - - - - - - - - - -")
print(f">> AUC is: {round(roc_auc_score(y_test, y_pred), 2)}")
print("- - - - - - - - - - - - - - - -")
print(f">> The confusion matrix is: {confusion_matrix(y_test, y_pred)}")