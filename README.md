# TROMPA Music Emotion Recognition

## Description

The idea of using active learning for music emotion recognition is to allow the classification models to improve with new annotations from particular users. In general, we implemented the strategy of query by committee in which N classification models are used to produce prediction probabilities of data instances which have not been annotated. In short, uncertainty sampling using entropy is used over the prediction probabilities of all classifiers, in order to measure the uncertainty produced by particular predictions: instances with low entropy are assumed to be the most informative, while low entropy highlights the least informative instances that should be annotated by our users.

## Installation

Clone this repository:

```
git clone https://github.com/juansgomez87/active-learning
cd active-learning
```

The TROMPA-MER system offers five functions: create a new user, extract features from audio, predict an emotion from the features, get songs to be annotated, and retrain a model for a particular user. This repository also offers a Dockerfile to build and run a container for each of these functions.

Before starting, two downloads are needed:

1. Download the Music-Enthusiasts data from [here](https://drive.google.com/file/d/1ZsAKCXgfqNOSyD58ZF1sVKjbQ3hWBfGf/view?usp=sharing), and extract all the files inside the `data` directory. Since this data is protected, please request access.

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZsAKCXgfqNOSyD58ZF1sVKjbQ3hWBfGf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZsAKCXgfqNOSyD58ZF1sVKjbQ3hWBfGf" -O trompa-me.zip && rm -rf /tmp/cookies.txt

```
2. The features used to train our models are the IS13 feature set which are obtained using the OpenSmile toolbox. Download [this compiled version](https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz) and extract it to the `active-learning` home directory. Otherwise, compiling the Docker container will do this automatically for you.

#### To build with Docker:

In order to build the Docker container use:

```
sudo docker build -t trompa-mer .
```

Then use the Docker file to start the Flask app:
```
sudo docker run -it -v "$(pwd)"/models:/code/models trompa-mer

```

#### To run with a virtual environment:

Create the virtual environment:
```
python3 -m venv trompa-venv
source trompa-venv/bin/activate
pip3 install -r requirements.txt
```

Copy the `sklearn.py` into the virtual environment library:
```
cp xgboost/sklearn.py trompa-venv/lib/python3.8/site-packages/xgboost
```

## Usage
#### Create a user model:

-   Input: user ID, mode ['mc', 'hc', 'mix', 'rand']
-   Output: user’s folder to save models

Usage:

```
python3 create_user.py -i USER_ID -m MODE
```

Example:

```
python3 create_user.py -i 827 -m mc
```

Docker:

```
sudo docker run -it -v $(pwd)/models:/code/models trompa-mer python3 create_user.py -i 827 -m mc
```

#### Extract features from audio:

-   Input: audio file (in wav or mp3)
-   Output: features from IS13 feature set (csv file)

Usage:

```
python3 extract_features.py -i INPUT_AUDIO_FILE -o OUTPUT_CSV_FILE
```

Example:

```
python3 extract_features.py -i ./test_audio/test.mp3 -o ./test_feats/test_mp3.csv
```

Docker:

```
sudo docker run -it -v $(pwd)/test_audio/test.wav:/test.wav -v $(pwd)/test_feats/:/outdir trompa-mer python3 extract_features.py -i /test.wav -o /outdir/test_wav.csv
```

#### Predict emotion:

-   Input: features from IS13 feature set (csv file), model to load
-   Output: JSON file with emotion predictions

Usage:

```
python3 predict_emotion.py -i CSV_FILE -o JSON_FILE -m PKL_MODEL_FILE
```

Example:

```
python3 predict_emotion.py -i ./test_feats/test_wav.csv -o ./test_predictions/test_wav.json -m ./models/pretrained/classifier_xgb.it_0.pkl
```

Docker:

```
sudo docker run -it -v $(pwd)/test_feats/test_wav.csv:/test_wav.csv -v $(pwd)/test_predictions/:/outdir trompa-mer python3 predict_emotion.py -i /test_wav.csv -o /outdir/test_wav.json -m default
```

#### Get annotation from CE:

-   Input: user ID and audio_object CE ID
-   Output: string { MW_ID : quadrant }

Usage:

```
python3 get_annotations.py -i USER_ID -ao AUDIO_OBJECT_UUID
```

Example:

```
python3 get_annotations.py -i 15479 -ao 130534fe-dc88-4adf-becc-99b9e7dda4ed
```

Docker:

```
sudo docker run -it -v $(pwd)/models:/code/models trompa-mer python3 get_annotations.py -i 15479 -ao 130534fe-dc88-4adf-becc-99b9e7dda4ed
```

#### Get songs to be annotated

_Requirement:_ all features from the TROMPA ME dataset have to be previously saved in the `path_to_data` folder in `settings.py`.

-   Input: user ID
-   Output: list of songs to be annotated

Usage:

```
python3 get_hard_tracks.py -i USER_ID -q NUM_TRACKS
```

Example:

```
python3 get_hard_tracks.py -i 827 -q 10
```

Docker:

```
sudo docker run -it -v $(pwd)/models:/code/models trompa-mer python3 get_hard_tracks.py -i 827 -q 10
```

#### Re-train model:

_Requirement:_ To run this code, get_hard_tracks must be ran previously.

-   Input: list of annotations from user X, and iteration number
-   Output: retrained model for user X.

Usage:

```
python3 retrain_model.py -i USER_ID -a ANNOTATIONS
```

Examples new_anno.json, new_anno_2.json, and new_anno_3.json:

```
python3 retrain_model.py -i 827 -a new_anno.json
python3 retrain_model.py -i 827 -a new_anno_2.json
python3 retrain_model.py -i 827 -a new_anno_3.json
```

This last example will show an error since the annotated tracks do not fit the tracks that were calculated with get_hard_tracks.

Docker:

```
sudo docker run -it -v $(pwd)/models:/code/models trompa-mer python3 retrain_model.py -i 827 -a new_anno.json
```

#### Pretraining a model

This requires access to the DEAM dataset.

Example:

```
python3 deam_classifier.py -cv 5 -m xgb
```

## Note:

A small change was made for the [Xgboost library](https://github.com/dmlc/xgboost/) in order to retrain the models. Xgboost is licenced under an Apache License 2.0. If you are running all the files locally, you can copy the `sklearn.py` file to the `/usr/local/lib/python3.8/site-packages/xgboost` directory where xgboost was installed.
If you use the Docker container, the file will be updated automatically.

Change in xgboost/sklearn.py Line 853:

```
        else:
            # added for active learning
            if xgb_model is None:
                self.classes_ = np.unique(y)
                self.n_classes_ = len(self.classes_)
                if not self.use_label_encoder and (
                        not np.array_equal(self.classes_, np.arange(self.n_classes_))):
                    raise ValueError(label_encoding_check_error)
```
