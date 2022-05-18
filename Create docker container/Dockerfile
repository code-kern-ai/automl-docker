FROM python:3.8.10
 
WORKDIR /automl
COPY . /automl
 
RUN pip install -r requirements.txt
 
ENTRYPOINT ["python"]
CMD ["reuse_model.py"]