# Import statements go here
from fastapi import FastAPI, responses
from pydantic import BaseModel
from configparser import ConfigParser
from util import get_model,get_encoder

from embedders.classification.contextual import TransformerSentenceEmbedder

# Instantiate fastapi app
api = FastAPI()


class Text(BaseModel):
    text: list


config = ConfigParser()
config.read("/automl/ml/config.ini")
model = config["Transformer_Model"]["model_used"]
use_encoder = config["Encoder"]["usage"]
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

    # Load model 
    model = get_model()

    # Create embeddings to make text usable by ml model
    embeddings = transformer.transform(corpus)

    predictions = model.predict(embeddings).tolist()

    probabilities = model.predict_proba(embeddings).tolist()
    probabilities_max = [round(max(i), 4) for i in probabilities]

    results = []
    if use_encoder == "True":
        encode = get_encoder()
        predictions_labels = encode.inverse_transform(predictions).tolist()

        for i, j in zip(predictions_labels, probabilities_max):
            results.append({"label": i, "confidence": j})

    else:
        for i, j in zip(predictions, probabilities_max):
            results.append({"label": i, "confidence": j})

    return results
