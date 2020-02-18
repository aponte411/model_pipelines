export PROJECT_ID=$PROJECT_ID
export IMAGE_NAME=$IMAGE_NAME
docker tag ${IMAGE_NAME} us.gcr.io/${PROJECT_ID}/${IMAGE_NAME}
docker push us.gcr.io/${PROJECT_ID}/${IMAGE_NAME}
