docker build -t docker-ml-wine .

docker run -d -p 8501:8501 --name docker-ml-wine-container docker-ml-wine

docker ps

docker kill <containerid>

docker rm -f docker-ml-wine-container

docker container ls

docker image ls

docker image rm docker-ml-wine

docker image ls