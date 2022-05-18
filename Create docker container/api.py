# Import statements go here
import uvicorn 
from fastapi import FastAPI
from pydantic import BaseModel

# Instantiate fastapi app
app = FastAPI()

class request_body(BaseModel):
    text : str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Kern automl tool."}

@app.post("/data-to-predict")
def predict(data: request_body):
    return data

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)