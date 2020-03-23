FROM python:3.6-slim

RUN pip install virtualenv
RUN virtualenv /env
ENV VIRTUAL_ENV -p python3.6 /env
ENV PATH /env/bin:$PATH

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

EXPOSE 50052

COPY engines.py /app/
COPY trainers.py /app/
COPY utils.py /app/
COPY deployments/api /app/

WORKDIR /app/
CMD ["python", "server.py"]
