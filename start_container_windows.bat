
docker build -t automl-container-backend  .

docker run ^
-p 7531:7531 ^
--name automl-container ^
--rm ^
automl-container-backend

docker logs automl-container -f
