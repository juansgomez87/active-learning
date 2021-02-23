FROM python:3.6.9

MAINTAINER Juan Sebastián Gómez Cañón

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

COPY xgboost/sklearn.py /usr/local/lib/python3.6/site-packages/xgboost/