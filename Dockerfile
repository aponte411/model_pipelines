FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && ln -s /usr/local/bin/python3 python
ADD requirements.txt .
RUN pip3 install -r requirements.txt

COPY models .
COPY webapp .

RUN export FOLD=0
RUN export MODEL=randomforest
RUN export MODEL_PATH=models/${MODEL}_${FOLD}_trained

ENTRYPOINT ["python3", "webapp/app.py"]
