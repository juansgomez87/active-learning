#!/usr/bin/env python3
"""
Emotion algorithm to to fine-tune on user annotations.


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import trompace.connection
from trompace.config import config
from trompace.queries.templates import format_filter_query, format_itemlist_query
import argparse
import pandas as pd
import sys

from settings import *

def get_mw_id(ce_id):
	df = pd.read_csv(map_mw_to_ce)
	try:
		mw_id = df.mw_id[df.ce_id == ce_id].item()
	except:
		print('Muziekweb id not found, upload track to CE!')
		sys.exit()
	return mw_id

def get_quadrant(aro_val, val_val):
	if aro_val > 0 and val_val > 0:
		quad = 'Q1'
	elif aro_val > 0 and val_val < 0:
		quad = 'Q2'
	elif aro_val < 0 and val_val < 0:
		quad = 'Q3'
	elif aro_val < 0 and val_val > 0:
		quad = 'Q4'
	return quad

def query_to_ce():
    dict_filt = {'creator': 'https://ilde.upf.edu/trompa/users/{}'.format(args.input_user),
                 'targetNode': {'target': {'identifier': args.audioobject}
                 }}
    ret_items = ['identifier', 'motivationNode { title }', 'bodyNode { ... on Rating { ratingValue } }']
    query = format_filter_query(queryname= 'Annotation',
    	                        args=dict_filt,
    	                        return_items_list=ret_items)

    
    response = trompace.connection.submit_query(query)

    t_list = ['arousalRating', 'valenceRating']

    for anno in response['data']['Annotation']:
    	if len(anno['motivationNode']) > 0 and anno['motivationNode'][0]['title'] == 'arousalRating':
    		aro = anno['bodyNode'][0]['ratingValue']
    	elif len(anno['motivationNode']) > 0 and anno['motivationNode'][0]['title'] == 'valenceRating':
    		val = anno['bodyNode'][0]['ratingValue']
    
    quad = get_quadrant(aro, val)

    return quad


if __name__ == "__main__":
    # usage: python3 get_annotations.py -i USER_ID -ao AUDIO_OBJECT_UUID 
    # example: python3 get_annotations.py -i 15479 -ao 130534fe-dc88-4adf-becc-99b9e7dda4ed 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_user',
                        help='Input user ID to load the models',
                        action='store',
                        required=True,
                        dest='input_user')
    parser.add_argument('-ao',
                        '--audioobject',
                        help='Select ',
                        action='store',
                        required=True,
                        dest='audioobject')
    args = parser.parse_args()

    config.load('trompace.ini')

    mw_id = get_mw_id(args.audioobject)

    quad = query_to_ce()

    print('{{ {}: {} }}'.format(mw_id, quad))


    # pdb.set_trace()
