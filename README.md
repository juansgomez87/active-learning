# TROMPA Music Emotion Recognition

## Description
The idea of using active learning for music emotion recognition is to allow the classification models to improve with new annotations from particular users. In general, we implemented the strategy of query by committee in which N classification models are used to produce prediction probabilities of data instances which have not been annotated. In short, uncertainty sampling using entropy is used over the prediction probabilities of all classifiers, in order to measure the uncertainty produced by particular predictions: instances with low entropy are assumed to be the most informative, while low entropy highlights the least informative instances that should be annotated by our users. 

## Usage
Clone this repository:
```
git clone https://github.com/juansgomez87/active-learning
cd active-learning
```

The TROMPA-MER system offers five functions: create a new user, extract features from audio, predict an emotion from the features, get songs to be annotated, and retrain a model for a particular user. This repository also offers a Dockerfile to build and run a container for each of these functions. 

Before starting, two downloads are needed:

1. Download the Music-Enthusiasts data from [here](https://drive.google.com/file/d/1ZsAKCXgfqNOSyD58ZF1sVKjbQ3hWBfGf/view?usp=sharing), and extract all the files inside the `data` directory. Since this data is protected, please request access.

2. The features used to train our models are the IS13 feature set which are obtained using the OpenSmile toolbox. Download [this compiled version](https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz) and extract it to the `active-learning` home directory. Otherwise, compiling the Docker container will do this automatically for you. 

#### To build with Docker:

In order to build the Docker container use:
```
sudo docker build -t trompa-mer .
```

#### Create a user model:
- Input: user ID
- Output: user’s folder to save models 

Usage: 
```
python3 create_user.py -i USER_ID
```
Example: 
```
python3 create_user.py -i 827
```
Docker:
```
sudo docker run -it -v $(pwd)/models:/code/models trompa-mer python3 create_user.py -i 827
```

#### Extract features from audio:
- Input: audio file (in wav or mp3)
- Output: features from IS13 feature set (csv file)

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
- Input: features from IS13 feature set (csv file), model to load
- Output: JSON file with emotion predictions

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

#### Get songs to be annotated
_Requirement:_ all features from the TROMPA ME dataset have to be previously saved in the `path_to_data` folder in `settings.py`. 
- Input: user ID
- Output: list of songs to be annotated

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
- Input: list of annotations from user X, and iteration number
- Output: retrained model for user X.

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
A small change was made for the Xgboost library in order to retrain the models. If you are running all the files locally, ou can copy the `sklearn.py` file to the `/usr/local/lib/python3.6/site-packages/xgboost` directory where xgboost was installed. 
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