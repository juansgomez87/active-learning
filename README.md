# TROMPA Music Emotion Recognition

## Usage
The TROMPA-MER system offers five functions:

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