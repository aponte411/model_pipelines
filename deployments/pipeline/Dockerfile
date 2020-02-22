FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && ln -s /usr/local/bin/python3 python
ADD requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ARG GOOGLE_APPLICATION_CREDENTIALS
ENV GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}

ARG MODEL_PATH
ENV MODEL_PATH=${MODEL_PATH}

CMD ["python3", "pipeline.py"]
