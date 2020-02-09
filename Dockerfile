FROM python:3.7-slim-buster as base

FROM base as build

ADD requirements.txt .
RUN pip install -r requirements.txt

FROM build as run

CMD ["python", "src/train.py"]