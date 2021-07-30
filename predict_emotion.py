#!/usr/bin/env python3
"""
Emotion algorithm to predict an emotion quadrant

Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
import joblib
import json
from collections import Counter

from sklearn.preprocessing import StandardScaler

from settings import *

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class EmotionPredictor():
    def __init__(self,
                 feats,
                 output,
                 model):
        """Constructor method
        """
        self.feats_file = feats
        self.out_f = output
        self.feats = pd.read_csv(self.feats_file, sep=';')
        self.seed = np.random.seed(1987)
        self.model = joblib.load(model)
        self.dict_class = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4'}


    def run(self):
        # pre-extracted feature sets (i.e. deam dataset)
        if 'pcm_fftMag_mfcc_sma_de[14]_amean' in self.feats.columns.tolist():
            X = self.dataset.loc[:, 'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
        # features extracted with this library
        elif 'mfcc_sma_de[14]_amean' in self.feats.columns.tolist():
            X = self.feats.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean']
        else:
            print('Something is wrong with the input features. Exiting!')
            sys.exit(0)

        feat_scale = True
        if feat_scale:
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns).to_numpy()

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        quads_pred = [self.dict_class[_] for _ in y_pred.tolist()]
        freq_pred = dict(Counter(quads_pred))

        json_out = {'input_feats': self.feats_file,
                    'output_probs': y_prob,
                    'mean_probs': np.mean(y_prob, axis=0),
                    'highest_prob_quad': self.dict_class[np.argmax(np.mean(y_prob, axis=0))],
                    'output_predictions': y_pred,
                    'freq_pred': freq_pred,
                    'mode_quad': max(freq_pred, key=freq_pred.get),
                    }
        with open(self.out_f, 'w') as f:
            json.dump(json_out, f, cls=NumpyArrayEncoder)
        print('Output file {} saved!'.format(self.out_f))


if __name__ == "__main__":
    # Usage
    # full use: python3 predict_emotion.py -i CSV_FILE -o JSON_FILE -m PKL_MODEL_FILE
    # example: python3 predict_emotion.py -i ./test_feats/test_wav.csv -o ./test_predictions/test_wav.json -m ./models/pretrained/classifier_xgb.it_0.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        help='Select input features to extract (csv file)',
                        action='store',
                        required=True,
                        dest='input')
    parser.add_argument('-o',
                        '--output',
                        help='Select output file with predictions (json file)',
                        action='store',
                        required=True,
                        dest='output')
    parser.add_argument('-m',
                        '--model',
                        help='Select model to generate predictions (pkl file)',
                        action='store',
                        required=True,
                        dest='model')
    args = parser.parse_args()

    if args.input.lower().endswith('.csv') is False:
        print('Input must be a csv file!')
        sys.exit(0)
    elif args.output.lower().endswith('.json') is False:
        print('Output must be a json file!')
        sys.exit(0)
    elif os.path.exists(args.input) is False:
        print('Select existing input file!')
        sys.exit(0)
    elif args.model == 'default':
        print('Selecting default model file!')
        args.model = './models/pretrained/classifier_xgb.it_0.pkl'
    elif os.path.exists(args.model) is False:
        print('Select existing model file!')
        sys.exit(0)


    emo_pred = EmotionPredictor(args.input, args.output, args.model)

    res = emo_pred.run()
