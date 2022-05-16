FROM python:3.8.10
 
WORKDIR /app
COPY . /app
 
RUN pip install -r requirements.txt
 
ENTRYPOINT ["python"]
CMD ["app.py"]