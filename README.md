# prac_5_sem

## Чтобы собрать и запустить docker-образ необходимо выполнить команды:

docker build -t ml_serv . \
docker run -p 5000:5000 --rm -i ml_serv

## Или можно скачать и запустить docker-образ так:

docker pull dmpopov1/task3:latest \
docker run -p 5000:5000 --rm -i dmpopov1/task3:latest

