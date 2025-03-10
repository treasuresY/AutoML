docker stop automl-front
docker rm automl-front

sleep 2
docker rmi registry.cn-hangzhou.aliyuncs.com/treasures/automl-frontend:20250120

sleep 1
docker build -t registry.cn-hangzhou.aliyuncs.com/treasures/automl-frontend:20250120 -f automl-frontend.Dockerfile .
docker run -itd --name automl-front -p 32000:8080 registry.cn-hangzhou.aliyuncs.com/treasures/automl-frontend:20250120

