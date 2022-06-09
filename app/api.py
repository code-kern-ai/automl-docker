# Import statements go here
import pickle
import numpy as np
from fastapi import FastAPI, responses
from pydantic import BaseModel
from configparser import ConfigParser

from embedders.classification.contextual import TransformerSentenceEmbedder

# Instantiate fastapi app
api = FastAPI()

class Text(BaseModel):
    text: list

config = ConfigParser()
config.read('/home/leonardpuettmann/repos/automl-docker/ml/config.ini')
model = config['Transformer_Model']['model_used']
transformer = TransformerSentenceEmbedder(model)

@api.get("/")
def root():
    return responses.RedirectResponse(url="/docs")

@api.post("/predict")  # response_model=Predictions
def predict(data: Text):
    """
    Get text data and returns predictions from a ml model.
    """

    # Load in the data, lower all words
    corpus = data.text

    # Create embeddings to make text usable by ml model
    embeddings = transformer.transform(corpus)

    # Use ml model to create predictions
    model = pickle.load(open("/home/leonardpuettmann/repos/automl-docker/ml/model.pkl", "rb"))
    predictions = model.predict(embeddings).tolist()

    predictions = model.predict(embeddings).tolist()
    probabilities = model.predict_proba(embeddings)
    probabilities_max = np.max(probabilities, axis=1).tolist()
    probabilities_rounded = [round(prob, 2) for prob in probabilities_max]
    probabilities_pct = [prob * 100 for prob in probabilities_rounded]
    
    dictionary = dict(zip(probabilities_pct, predictions))
    return dictionary