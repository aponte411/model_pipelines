FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && ln -s /usr/local/bin/python3 python
ADD requirements.txt .
RUN pip3 install -r requirements.txt

COPY pipeline/pipeline.py .
COPY pipeline/pipeline-creds.json .

RUN export GOOGLE_APPLICATION_CREDENTIALS=/pipeline-creds.json
RUN export MODEL_PATH=pipeline_run

ENTRYPOINT ["python3", "pipeline.py"]
