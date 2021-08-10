#!/usr/bin/env python3
"""
Emotion algorithm feature extractor


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import os
import sys
import argparse
import shutil
import time
import json
import pandas as pd
import subprocess
import numpy as np
from collections import Counter
from dotenv import load_dotenv

import pdb

from settings import *


def load_json(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = json.loads(data)
    return data

def get_quadrant(arousal, valence):
    if arousal > 0 and valence > 0:
        quad = 'Q1'
    elif arousal > 0 and valence < 0:
        quad = 'Q2'
    elif arousal < 0 and valence < 0:
        quad = 'Q3'
    elif arousal < 0 and valence > 0:
        quad = 'Q4'
    return quad

def consensus_hc(dataset_anno, u_id):
    # load annotations
    data = load_json(dataset_anno)
    anno = pd.DataFrame(data['annotations'])
    users = pd.DataFrame(data['users'])
    anno.rename(columns={'externalID': 's_id'}, inplace=True)
    anno['s_id'] = anno['s_id'].map(lambda x: x.lower())
    # calculate quadrants
    aro_list = anno.arousalValue.values.astype(int)
    val_list = anno.valenceValue.values.astype(int)
    quad_list = list(map(get_quadrant, aro_list, val_list))
    anno['quadrant'] = quad_list

    # remove songs already annotated by this user
    anno = anno[anno.userid != str(u_id)]

    # calculate frequencies for human consensus entropy
    frequencies = {}
    for id_song in anno.s_id.unique().tolist():
        cnt_quad = Counter({'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0})
        cnt_quad.update(anno[anno.s_id == id_song].quadrant)
        num_anno = anno.loc[anno.s_id == id_song].shape[0]
        cnt_quad = dict(cnt_quad)
        frequencies[id_song] = {k: np.round(v / num_anno, 3) for k, v in cnt_quad.items()}
    
    freqs = pd.DataFrame(frequencies).transpose()
    freqs.index.name = 's_id'
    return freqs

def create_user(u_id, mode):
    # create users folder
    user_path = os.path.join(path_models_users, str(u_id), mode)
    try:
        os.makedirs(user_path)
    except FileExistsError:
        print('User has already been created, exiting!')
        # subprocess.run(['rm', '-rf', user_path])
        # os.makedirs(user_path)
        sys.exit()

    # copy initial models
    pre_models = [os.path.join(root, f) for root, dirs, files in os.walk(path_models)
                  for f in files if (f.lower().endswith('.pkl') and 
                  os.path.join(root, f).count(os.path.sep) <= path_models.count(os.path.sep) + 1)]
    cp_models = [f.replace(path_models, user_path) for f in pre_models]

    for in_f, out_f in zip(pre_models, cp_models):
        shutil.copy(in_f, out_f)

    freqs = consensus_hc(dataset_anno, u_id)
    freqs['log'] = -1
    freqs_fn = os.path.join(user_path, 'consensus.csv')
    freqs.to_csv(freqs_fn)

    print('User {} created!'.format(u_id))


if __name__ == "__main__":
    # usage: python3 create_user.py -i USER_ID
    # example: python3 create_user.py -i 827
    # average time: Process lasted 0.899834394454956 seconds!
    load_dotenv()

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_user',
                        help='Input user ID to create model and copy baseline classifiers',
                        action='store',
                        required=True,
                        dest='input_user')
    parser.add_argument('-m',
                        '--mode',
                        help='Select mode of function: machine-consensus [mc], human consensus [hc], hybrid [mix], or random [rand]',
                        action='store',
                        required=True,
                        dest='mode')
    args = parser.parse_args()

    try:
        # TODO: user ids are created as numbers, will this change?
        user_id = int(args.input_user)
    except ValueError:
        print('User ID is invalid')
        sys.exit(0)

    if args.mode != 'hc' and args.mode != 'mc' and args.mode != 'mix' and args.mode != 'rand':
        print('Select a valid consensus calculation mode!')
        sys.exit()

    create_user(user_id, args.mode)
    print('Process lasted {} seconds!'.format((time.time()-start)))
