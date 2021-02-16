#!/usr/bin/env python3
"""
Settings


Copyright 2021, J.S. Gómez-Cañón
Licensed under ???
"""
path_models = './models/pretrained'
path_models_users = './models/users'
path_to_data = './data'
# trompa data
path_to_audio = '{}/audio'.format(path_to_data)
path_to_feats = '{}/feats'.format(path_to_data)
dataset_fn = '{}/dataset_feats.csv'.format(path_to_data)
dataset_anno = '{}/data_04_11_2020.json'.format(path_to_data)

# deam data
# this is only needed to pretrain the models
deam_feats = '/media/hoodoochild/DATA/datasets/deam/features'
deam_dataset_fn = '/media/hoodoochild/DATA/datasets/deam/dataset_quads.csv'
deam_anno_arousal = 'deam_annotations/arousal.csv'
deam_anno_valence = 'deam_annotations/valence.csv'
