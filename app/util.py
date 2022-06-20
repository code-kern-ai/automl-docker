
import pickle

__loaded_model = None
__loaded_encoder = None

def get_model():
    global __loaded_model
    if not __loaded_model:
        __loaded_model = pickle.load(open("/automl//ml/model.pkl", "rb"))
    return __loaded_model

def get_encoder():
    global __loaded_encoder
    if not __loaded_encoder:
        __loaded_encoder = pickle.load(open("/automl//ml/encoder.pkl", "rb"))
    return __loaded_encoder
