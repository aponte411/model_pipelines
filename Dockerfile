FROM python:3.7-slim-buster as base

ADD . .
RUN pip install -r requirements.txt

CMD ["python", "src/train.py"]