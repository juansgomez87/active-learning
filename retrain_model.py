#!/usr/bin/env python3
"""
Emotion algorithm to fine-tune on user annotations.


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import argparse
import numpy as np
import pandas as pd
import json
import click
import os
import sys
import pdb
import joblib
from scipy.stats import entropy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

from settings import *


class Retrainer():
    def __init__(self,
                 anno_dict,
                 path_to_models):
        """Constructor method
        """
        anno_dict = self.load_json(anno_dict)
        self.anno_dict = {k.lower(): v for k, v in anno_dict.items()}
        self.path_to_feats = path_to_feats
        mode = [d for root, dirs, files in os.walk(path_to_models) for d in dirs][0]
        self.out_f = os.path.join(path_to_models, mode, 'pool_data.json')
        self.mod_list = [os.path.join(root, f) for root, dirs, files in os.walk(path_to_models) for f in files if f.lower().endswith('.pkl')]
        self.dict_class = {0: 'Q1', 1: 'Q2', 2: 'Q3', 3: 'Q4'}


        if os.path.exists(dataset_fn):
            self.dataset = pd.read_csv(dataset_fn, sep=';')
        else:
            self.dataset = self.load_feats(dataset_fn)

        print(self.anno_dict)

        X_train = StandardScaler().fit_transform(self.dataset.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean'])
        X_train = pd.DataFrame(X_train, index=self.dataset.s_id)
        self.X_train = X_train.loc[self.anno_dict.keys()]
        y_train = pd.DataFrame.from_dict(self.anno_dict, orient='index').reindex(self.X_train.index)
        self.y_train = LabelEncoder().fit_transform(y_train.values.ravel())

        if os.path.exists(self.out_f):
            self.pool_info = self.load_json(self.out_f)
            self.it_num = self.pool_info['iteration']
            len_query = len(self.pool_info['queried'])
            len_anno = len(self.anno_dict)
            set_pool = set(self.pool_info['queried'])
            set_anno = set(list(self.anno_dict.keys()))
            if len_anno != len_query:
                print('Number of annotations does not fit number of queries!')
                sys.exit()
            elif set_pool != set_anno:
                print('Input annotations do not fit get_hard_tracks output!')
                sys.exit()
            self.pool_info['log'].append({self.it_num: self.anno_dict})

            # self.old_recs = self.pool_info['recs_log']

            with open(self.out_f, 'w') as f:
                json.dump(self.pool_info, f, indent=4)

        else:
            print('Pooling file not created, run get_hard_tracks.py first!')
            sys.exit(0)
        # print('Current iteration number: {}'.format(self.it_num))


    def load_json(self, filename):
        with open(filename, 'r') as f:
            data = f.read()
        data = json.loads(data)
        return data


    def load_feats(self, dataset_fn):
        fill_char = click.style('=', fg='yellow')
        feats_list = [os.path.join(root, f) for root, dirs, files in os.walk(self.path_to_feats) for f in files if f.lower().endswith('.csv')]
        id_list = [_.split('/')[-1].replace('.csv', '').replace('-sample', '') for _ in feats_list]

        all_feats = []
        with click.progressbar(range(len(id_list)), label='Loading features and processing...', fill_char=fill_char) as bar:
            for feat, s_id, i in zip(feats_list, id_list, bar):
                this_feat = pd.read_csv(feat, sep=';')
                this_feat['s_id'] = s_id
                del this_feat['frameTime']
                all_feats.append(this_feat)

        df_all_feats = pd.concat(all_feats, axis=0)
        df_all_feats.reset_index().drop(columns='index', inplace=True)
        df_all_feats.to_csv(dataset_fn, sep=';', index=False)
        return df_all_feats


    def run(self):
        # re train each model
        for i, mod_fn in enumerate(self.mod_list):
            # print('Performing retraining for model {} ({}/{})'.format(mod_fn, i, len(self.mod_list) - 1))
            mod = joblib.load(mod_fn)
            if mod_fn.find('_xgb') > 0:
                # TODO: check metric evaluation
                # print('Extreme gradient boosting model')
                mod.fit(self.X_train.values, self.y_train, eval_metric='auc', xgb_model=mod.get_booster()) 
            else:
                # print('Gaussian naive bayes model')
                mod.partial_fit(self.X_train.values, self.y_train)
            joblib.dump(mod, mod_fn)
        




if __name__ == "__main__":
    # usage: python3 retrain_model.py -i USER_ID -a ANNOTATIONS
    # example: python3 retrain_model.py -i 827 -a new_anno.json
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_user',
                        help='Input user ID to load the models',
                        action='store',
                        required=True,
                        dest='input_user')
    parser.add_argument('-a',
                        '--annotations',
                        help='Input annotations for this iteration (json file)',
                        action='store',
                        required=True,
                        dest='annotations')

    args = parser.parse_args()

    try:
        user_id = int(args.input_user)
        path_to_models = os.path.join(path_models_users, str(user_id))
        if os.path.exists(path_to_models) is False:
            print('This user has not been created yet!')
            sys.exit(0)
    except ValueError:
        print('User ID is invalid')
        sys.exit(0)

    if os.path.exists(args.annotations) is False:
        print('Select existing input annotations file!')
        sys.exit(0)
 
    retrain = Retrainer(args.annotations, path_to_models)

    retrain.run()

    # print('Process lasted {} seconds!'.format((time.time()-start)))


