kubectl create secret generic airflow-postgres --from-literal=postgres-password=$(openssl rand -base64 13)
kubectl create secret generic airflow-redis --from-literal=redis-password=$(openssl rand -base64 13)
