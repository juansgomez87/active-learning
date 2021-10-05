FROM python:3.8.5

LABEL maintainer="juansebastian.gomez@upf.edu"

# Common requirements
RUN apt-get update \
    && apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /code
WORKDIR /code

# install opensmile
# RUN curl -SL https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz \
#    | tar -xvz 
#RUN chmod 755 opensmile-3.0-linux-x64/bin/SMILExtract


RUN pip3 install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache -r /tmp/requirements.txt

# Small modification to allow retraining with incomplete classes. See note in documentation.
COPY xgboost/sklearn.py /usr/local/lib/python3.8/site-packages/xgboost/
#COPY src/xgboost/sklearn.py /usr/local/lib/python3.8/site-packages/xgboost/

COPY . /code
#COPY ./src /code/src/

EXPOSE 5000

CMD ["python3", "/code/flask_service.py"]
#CMD ["python3", "/code/src/flask_service.py"]