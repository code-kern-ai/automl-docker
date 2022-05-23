FROM python:3.8.10
 
WORKDIR /automl
COPY . /automl
 
RUN pip install -r requirements.txt

# Run api and models
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]


