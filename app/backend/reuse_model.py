# Basis libraries
import pickle

# The functions below can be used to load ml models and transformers to pre-process texts.
# Loading transfomer from pickled file. We need to use a pickled transformer that was "learned" on an existing text corpus to be able to transform new texts.
def load_transformer():
    """
    Loads in a pickled transformer model. Note that the transformer must be created
    using the 'create_model.py' file.
    """

    transformer = pickle.load(open("ml/transformer.pk", "rb"))
    return transformer


# Use previously loaded transformer to create word embeddings.
def create_embeddings(transformer, corpus):
    """
    Uses loaded transformer to create word embeddings.

    Args:
    - transformer -> Loaded transformer model from the 'load transformer' function.
    - corpus -> Text corpus
    """

    embeddings = transformer.transform(corpus)
    return embeddings


# Load model from a pickle file.
def create_predictions(embeddings):
    """
    Creates predictions with either an already trained neural net or a
    logistic regression.

    Args:
    - embeddings -> Word embeddings created by the 'create_embeddings' function.
    """

    model = pickle.load(open("ml/Logistic Regression.pkl", "rb"))
    preds = model.predict(embeddings)
    return preds
