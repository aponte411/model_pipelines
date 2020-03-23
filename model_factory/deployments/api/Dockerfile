FROM python:3.6-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 50052

COPY deployments/api /root/
WORKDIR /root
CMD ["python", "server.py"]
