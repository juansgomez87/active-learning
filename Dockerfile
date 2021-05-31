#FROM python:3.6.9
FROM python:3.8.5


LABEL maintainer="juansebastian.gomez@upf.edu"

# Common requirements
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    python3 \
    ffmpeg \
    libsndfile1-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN mkdir /code
COPY . /code
WORKDIR /code

RUN curl -SL https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz \
    | tar -xvz 

RUN chmod 755 /code/opensmile-3.0-linux-x64/bin/SMILExtract

# Small modification to allow retraining with incomplete classes. See note in documentation.

COPY xgboost/sklearn.py /usr/local/lib/python3.6/site-packages/xgboost/