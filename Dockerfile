FROM python:3.6.9

# Common requirements
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    python3 \
    ffmpeg \
    libsndfile1-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN mkdir /code
COPY . /code
WORKDIR /code

CMD ["/code/create_user.py", "/code/extract_features.py", "/code/predict_emotion.py", "/code/get_hard_tracks.py", "/code/retrain_model.py"]

ENTRYPOINT ["python3"]
