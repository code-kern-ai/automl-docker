# Import statements go here
from pyexpat import features
import uvicorn 
from fastapi import FastAPI

from pydantic import BaseModel

# Instantiate fastapi app
app = FastAPI()

class Data(BaseModel):
    features: str
    labels: int

@app.post('/data')
def get_data(data: Data):
    features = data['features']
    labels = data['labels']
    return features, labels

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)