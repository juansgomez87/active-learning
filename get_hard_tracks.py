#!/usr/bin/env python3
"""
Emotion algorithm to to fine-tune on user annotations.


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
from sklearn.preprocessing import StandardScaler

from trompace.config import config
from trompace.client.client_itemlist import create_itemlist


from settings import *


class ConsensusEntropyCalculator():
    def __init__(self,
                 queries,
                 path_to_models):
        """Constructor method
        """
        self.queries = queries
        self.path_to_feats = path_to_feats
        self.out_f = os.path.join(path_to_models, 'pool_data.json')
        self.mod_list = [os.path.join(root, f) for root, dirs, files in os.walk(path_to_models) for f in files if f.lower().endswith('.pkl')]

        if os.path.exists(dataset_fn):
            self.dataset = pd.read_csv(dataset_fn, sep=';')
        else:
            self.dataset = self.load_feats(dataset_fn)

        X_pool = StandardScaler().fit_transform(self.dataset.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean'])
        self.X_pool = pd.DataFrame(X_pool, index=self.dataset.s_id)

        if os.path.exists(self.out_f):
            self.pool_info = self.load_json(self.out_f)
            self.it_num = self.pool_info['iteration'] + 1
            self.X_pool = self.X_pool.loc[self.pool_info['next_pool']]
        else:
            self.it_num = 0
        print('Iteration number: {}'.format(self.it_num))


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

    def send_itemlist(self, it_l, u_id):
        name = 'get_hard_tracks user {}'.format(u_id)
        description = 'get_hard_tracks for MER'
        ordered = True
        contributor ='MTG - MER algorithm'
        creator = 'Juan S. Gómez'
        create_itemlist(name=name,
                        description=description,
                        ordered=ordered,
                        contributor=contributor,
                        creator=creator,
                        values=it_l)


    def run(self):
        # predict probabilities for each model
        pred_prob = []

        for i, mod_fn in enumerate(self.mod_list):
            print('Predicting probabilities for model {} ({}/{})'.format(mod_fn, i, len(self.mod_list) - 1))
            mod = joblib.load(mod_fn)
            y_probs = mod.predict_proba(self.X_pool)
            # summarize with mean across all samples
            y_probs = pd.DataFrame(y_probs, index=self.X_pool.index).groupby(['s_id']).mean()
            pred_prob.append(y_probs)

        # disagreement-based consensus entropy
        consensus_prob = np.mean(np.array(pred_prob), axis=0)
        ent = entropy(consensus_prob, axis=1)
        # select songs with max entropy for self.queries amount
        q_ind = np.argsort(ent)[::-1]
        full_q_songs = y_probs.iloc[q_ind].index.tolist()
        q_songs = full_q_songs[:self.queries]
        next_pool = full_q_songs[self.queries:]
        # save json with info
        json_out = {'iteration': self.it_num,
                     'next_pool': next_pool,
                     'queried': q_songs}
        with open(self.out_f, 'w') as f:
            json.dump(json_out, f)
        return q_songs


if __name__ == "__main__":
    # usage: python3 get_hard_tracks.py -i USER_ID -q NUM_TRACKS 
    # example: python3 get_hard_tracks.py -i 827 -q 10 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_user',
                        help='Input user ID to load the models',
                        action='store',
                        required=True,
                        dest='input_user')
    parser.add_argument('-q',
                        '--queries',
                        help='Select number of queries to perform (int)',
                        action='store',
                        type=int,
                        required=True,
                        dest='queries')

    args = parser.parse_args()


    try:
        # TODO: user ids are created as numbers, will this change?
        user_id = int(args.input_user)
        path_to_models = os.path.join(path_models_users, str(user_id))
        if os.path.exists(path_to_models) is False:
            print('This user has not been created yet!')
            sys.exit(0)
    except ValueError:
        print('User ID is invalid')
        sys.exit(0)


    config.load('trompace.ini')

    ce_en_cal = ConsensusEntropyCalculator(args.queries, path_to_models)

    q_songs = ce_en_cal.run()

    ce_en_cal.send_itemlist(q_songs, user_id)

    print('Songs to annotate: {}'.format(q_songs))

