# Import statements go here
from fastapi import FastAPI, responses
from pydantic import BaseModel
import pickle

from embedders.classification.contextual import TransformerSentenceEmbedder

# Instantiate fastapi app
api = FastAPI()


class Text(BaseModel):
    text: list


transformer = TransformerSentenceEmbedder("distilbert-base-uncased")


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
    model = pickle.load(open("ml/Logistic Regression.pkl", "rb"))
    return model.predict(embeddings).tolist()
