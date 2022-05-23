# Import statements go here
from fastapi import FastAPI
from pydantic import BaseModel
from reuse_model import (
    load_transformer,
    create_embeddings,
    create_predictions,
)

# Instantiate fastapi app
api = FastAPI()


class Text(BaseModel):
    text: list


# class Predictions(BaseModel):
#     preds: dict


@api.get("/")
def read_root():
    """
    Simple greeting function.
    """
    return {"message": "Welcome to the Kern automl tool."}


@api.post("/predict")  # response_model=Predictions
def predict(data: Text):
    """
    Get text data and returns predictions from a ml model.
    """

    # Load in the data, lower all words
    corpus = data.text

    # Load transformer model from current directory
    transformer = load_transformer()

    # Create embeddings to make text usable by ml model
    embeddings = create_embeddings(transformer, corpus)

    # Use ml model to create predictions
    prediction = create_predictions(embeddings)
    return prediction.tolist()
