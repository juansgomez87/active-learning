# TROMPA Music Emotion Recognition

## Description
The idea of using active learning for music emotion recognition is to allow the classification models to improve with new annotations from particular users. In general, we implemented the strategy of query by committee in which 5 classification models are used to produce prediction probabilities of data instances which have not been annotated. In short, uncertainty sampling using entropy is used over the prediction probabilities of all classifiers, in order to measure the uncertainty produced by particular predictions: instances with low entropy are assumed to be the most informative, while low entropy highlights the least informative instances that should be annotated by our users. 

## Usage
The TROMPA-MER system offers five functions: create a new user, extract features from audio, predict an emotion from the features, get songs to be annotated, and retrain a model for a particular user. To start download the data from [here](https://drive.google.com/file/d/1ZsAKCXgfqNOSyD58ZF1sVKjbQ3hWBfGf/view?usp=sharing), and extract all the files inside the `data` directory. 

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
sudo docker run -it -v $(pwd)/test_feats/test_wav.csv:/test_wav.csv -v $(pwd)/test_predictions/:/outdir trompa-mer python3 predict_emotion.py -i /test_wav.csv -o /outdir/test_wav.json -m ./models/pretrained/classifier_xgb.it_0.pkl
```

#### Get songs to be annotated
Requirement: all features from the data set have to be previously extracted and saved in the `path_to_data` folder in `settings.py`.
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
- Input: list of annotations from user X, and iteration number
- Output: retrained model for user X.

Usage: 
```
python3 retrain_model.py -i USER_ID -a ANNOTATIONS
```
Example:
```
python3 retrain_model.py -i 827 -a new_anno.json
```
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