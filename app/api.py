# Import statements go here
from fastapi import FastAPI, responses
from pydantic import BaseModel
import pickle
from configparser import ConfigParser

from embedders.classification.contextual import TransformerSentenceEmbedder

# Instantiate fastapi app
api = FastAPI()

class Text(BaseModel):
    text: list

config = ConfigParser()
config.read('ml/config.ini')
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
    model = pickle.load(open("ml/model.pkl", "rb"))
    return model.predict(embeddings).tolist(), model.predict_proba(embeddings).tolist()
