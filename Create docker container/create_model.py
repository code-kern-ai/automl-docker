# Filter all debugging messages from tensorflow 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, mean_squared_error

# Embedders and Transformers
from sentence_transformers import SentenceTransformer

# XGBoost
import xgboost as xgb

# Tensorflow 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print(' ')
print('  ___________      _  __                          _____ ') 
print(' |  _______  |    | |/ /                    /\   |_   _|') 
print(' | |       | |    |   / ___ _ __ _ __      /  \    | | ') 
print('| |        | |    |  < / _ \ |__| |_ \    / /\ \   | | ')
print('| |_______| |     | . \  __/ |  | | | |  / ____ \ _| |_ ') 
print('|___________|     |_|\_\___|_|  |_| |_| /_/    \_\_____|') 
print(' ')
print('AutoML Tool by kern.ai ©')
print('Please visit https://github.com/code-kern-ai/automl-docker for instructions.')
print(' ')

def input_getter(path_statement):
    while True:
        user_input = input()
        print(path_statement, user_input)
        path_approval = input('(y/ n) ')
        if path_approval.lower() == 'y':
            break
        elif path_approval.lower() == 'n':
            print('>> Enter a new path: ')
        else:
            print(">> Sorry, that didn't work. Please enter again: ")
    return user_input

# get datetime dd/mm/YY H:M
now = datetime.now()
dt_string = now.strftime('%d-%m-%Y %H-%M')

# Read in the data with pandas, then convert text corpus to list
print('>> Please select the path to where your data is stored!')
print(' ')
print('>> On Windows the path might look like this  ->  C:\\Users\\yourname\\data\\training_data.csv')
print('>> On MacOS/ Linux the path might look like this  ->  home/user/data/training_data.csv')
print(' ')

# Get the path where the data is stored
PATH = input_getter('>> Is this the correct path? ->')
df = pd.read_csv(PATH)
print('>> Data successfully loaded!')

# Get the name of the features
print(' ')
print('>> Please provide the column name in which the texts are store in!')
COL_TEXTS = input_getter('>> Is this the correct column name? ->')

# Load the data with the provided info, lower all words in the corpus
corpus = df[COL_TEXTS].to_list()
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()

# Get the names of the labels
print(' ')
print('>> Please provide the column name in which the labels are store in!')
COL_LABEL = input_getter('>> Is this the correct column name? ->')

while True:
    print(' ')
    print('>> Please input a number to choose your method of preprocessing the text data.')
    print('>> 1 - Very accurate, state of the art method, but slow (especially on large datasets)')
    print('>> 2 - Fast but less accurate (but still good)')
    print(' ')

    choice = input()

    if choice == '1':
        # Instantiate a sentence transformer and create embeddings 
        print('>> Creating embeddings using transformer model, this might take a couple of minutes ...')
        sent_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = sent_transformer.encode(corpus)

        # Pickle transformer to be reusable in a different file
        with open('transformer.pk', 'wb') as fin:
            pickle.dump(sent_transformer, fin)
        break

    else:
        # Intantiate a tf-idf vectorizer
        print('>> Creating TF-IDF embeddings ...')
        vect = TfidfVectorizer()
        embeddings = vect.fit_transform(corpus)

        # Pickle vectorizer to be reusable in a different file
        with open('transformer.pk', 'wb') as fin:
            pickle.dump(vect, fin)
        break

features = embeddings

labels = df[COL_LABEL]
labels = np.array(labels)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Selecting a model
while True:
    print(' ')
    print('>> Please input a number to choose your algorithm:')
    print('>> 1 - Logistic Regression')
    print('>> 2 - Random Forest Classifier')
    print('>> 3 - XGBoost Classifier')
    print('>> 4 - Noise Robust Deep Neural Network (work in progress)')
    print(' ')

    model_choice = input()

    if model_choice == '1':
        print('>> Training a logistic regression ...')
        lr = LogisticRegression(dual=False)

        # Hyper parameter space is relatively small
        hyperparameters = {'C': np.arange(0, 4), 
                        'penalty': ['l2', 'None'],
                        'max_iter': [100, 150, 250, 500]}

        # Initiate and fit random search cv
        lr_clf = RandomizedSearchCV(estimator=lr, param_distributions=hyperparameters, cv=5,  n_iter=5)
        lr_clf.fit(X_train, y_train)
        y_pred = lr_clf.predict(X_test)

        # Save the model to current directory
        with open(f'Logistic Regression {dt_string}.pkl', 'wb') as fid:
            pickle.dump(lr_clf, fid) 
            print(f'>> Saved model to {os.path.abspath(os.getcwd())}') 
        break

    elif model_choice == '2':
        print('Training a random forest ...')
        rf = RandomForestClassifier()

        # Hyper parameter space is relatively small
        hyperparameters = {'max_depth': [90, 100, 110], 
                        'min_samples_leaf': [3, 4, 5]}

        # Initiate and fit random search cv
        rf_clf = RandomizedSearchCV(estimator=rf, param_distributions=hyperparameters, cv=3, n_iter=3)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)

        # Save the model to current directory
        with open(f'Random Forest {dt_string}.pkl', 'wb') as fid:
            pickle.dump(rf_clf, fid) 
            print(f'>> Saved model to {os.path.abspath(os.getcwd())}') 
        break

    elif model_choice == '3':
        print('Training a XGBoost classifier ...')
        xg_cl = xgb.XGBClassifier(eval_metric='auc')

        # Hyperparameter space is relatively narrow
        hyperparameters = {'eta': np.arange(0.1, 0.5, 0.1), 
                        'max_depth': [4, 6, 8, 10], 
                        'n_estimators': np.arange(10, 400, 20), }

        # Initiate and fit random search cv
        xgb_clf = RandomizedSearchCV(estimator=xg_cl, param_distributions=hyperparameters, cv=3, n_iter=3)
        xgb_clf.fit(X_train, y_train)
        y_pred = xgb_clf.predict(X_test)

        # Save the model to current directory
        with open(f'XGBoost {dt_string}.pkl', 'wb') as fid:
            pickle.dump(xgb_clf, fid) 
            print(f'>> Saved model to {os.path.abspath(os.getcwd())}') 
        break

    elif model_choice == '4':
        print('Training neural network ...')
        
        #Convert Training and testing data to float16
        X_train.astype('float16')
        X_test.astype('float16')
        y_train.astype('float16')

        # Vectorize labels
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

        # Implement early stopping to prevent overfitting
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        # Setup the neural network
        neural_net = Sequential()
        neural_net.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], )))
        neural_net.add(Dropout(0.2))
        neural_net.add(Dense(64, activation='relu'))
        neural_net.add(Dropout(0.2))
        neural_net.add(Dense(32, activation='relu'))
        neural_net.add(Dense(1, activation='sigmoid'))

        neural_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Get model params as a summary
        #neural_net.summary()

        # Finally, train the model
        neural_net.fit(X_train, y_train, epochs=5, verbose=True, callbacks=[callback])

        # Saveing, to be able to reuse the model later
        neural_net.save(f'Neural Net {dt_string}.h5')
        print(f'>> Saved model to {os.path.abspath(os.getcwd())}')
        print(' ')

        # Make predictions on the testing data
        y_pred = (neural_net.predict(X_test) > 0.5).astype("int32")
        break

    else:
        pass

print(' ')
print('>> Generating evaluation metrics on unseen testing data...')
print(' ')
print('- - - - - - - - - - - - - - - -')
print(f'>> Model accuracy is: {round(accuracy_score(y_test, y_pred), 2) * 100} %')
print('- - - - - - - - - - - - - - - -')
print(f'>> Mean squared error is: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)}')
print('- - - - - - - - - - - - - - - -')
print(f'>> AUC is: {round(roc_auc_score(y_test, y_pred), 2)}')
print('- - - - - - - - - - - - - - - -')
print(f'>> The confusion matrix is: {confusion_matrix(y_test, y_pred)}')