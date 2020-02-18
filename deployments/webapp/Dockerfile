FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && ln -s /usr/local/bin/python3 python
ADD requirements.txt .
RUN pip3 install -r requirements.txt

COPY trained_models/randomforest_0_trained .
COPY webapp/app.py .

CMD ["gunicorn", "--bind", "0.0.0.0", "wsgi:application"]
