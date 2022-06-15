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
use_encoder = config['Encoder']['usage']
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
    probabilities = model.predict_proba(embeddings)
    probabilities_max = np.max(probabilities, axis=1).tolist()
    probabilities_rounded = [round(prob, 4) for prob in probabilities_max]
    probabilities_pct = [prob * 100 for prob in probabilities_rounded]

    results = []
    if use_encoder == 'True':
        encode = pickle.load(open("/home/leonardpuettmann/repos/automl-docker/ml/encoder.pkl", "rb"))
        predictions_labels = encode.inverse_transform(predictions).tolist()

        for i, j in zip(predictions_labels, probabilities_pct):
            results.append({'label': i, 'confidence': j})   
    else:
        for i, j in zip(predictions, probabilities_pct):
            results.append({'label': i, 'confidence': j})
    return results