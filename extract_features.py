#!/usr/bin/env python3
"""
Emotion algorithm feature extractor


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import numpy as np
import pandas as pd
import os
import sys
import subprocess
import argparse

import pdb

def extract_features(in_f, out_f):
    if in_f.lower().endswith('.mp3'):
        tmp_file = out_f.replace('.csv', '.wav')
        # convert from mp3 to wav
        subprocess.run(['ffmpeg', '-v', 'quiet', '-i', in_f, tmp_file])
    elif in_f.lower().endswith('.wav'):
        tmp_file = in_f
    else:
        print('Input file must be mp3 or wav!')
        sys.exit(0)

    # process with open smile INTERSPEECH 2013 ComParE
    subprocess.run(['./opensmile-3.0-linux-x64/bin/SMILExtract',
                 '-C',
                 './opensmile-3.0-linux-x64/config/is09-13/IS13_ComParE.conf',
                 '-I',
                 tmp_file,
                 '-lldcsvoutput',
                 out_f,
                 '-noconsoleoutput',
                 '1'])

    # calculate mean and standard dev on a new data drame (1 sample = 10ms)
    # 1000 ms, 50% overlap: chunk = 100, n_overlap = 500
    # 500 ms, 0% overlap: chunk = 50, n_overlap = 1
    chunk = 100
    n_overlap = int(chunk * 0.5)
    d_f = pd.read_csv(out_f, sep=';').drop('name', axis=1)

    idx_list = d_f.columns.tolist()
    new_idx_list = [idx_list[0]]
    for i in range(1, len(idx_list)):
        new_idx_list.append(idx_list[i]+'_stddev')
        new_idx_list.append(idx_list[i]+'_amean')
    idx = [_ for _ in range(int(d_f.shape[0]/chunk) + 1)]
    new_d_f = pd.DataFrame(index=idx, columns=new_idx_list)

    # in case of overlap
    for idx, row in enumerate(range(0, d_f.shape[0], n_overlap)):
        new_d_f.loc[idx, 'frameTime'] = np.round(d_f.loc[row, 'frameTime'], decimals=2)
        for this_key in d_f.drop(columns='frameTime').columns:
            #this_key = d_f.columns.tolist()[col]
            this_key_mean = this_key + '_amean'
            this_key_std = this_key + '_stddev'
            
            new_d_f.loc[idx, this_key_mean] = np.mean(d_f.loc[row:row+chunk, this_key])
            new_d_f.loc[idx, this_key_std] = np.std(d_f.loc[row:row+chunk, this_key])
    new_d_f.to_csv(out_f, sep=';', index=False)

    if in_f.lower().endswith('.mp3'):
        subprocess.run(['rm', tmp_file])

    print('Output file {} created!'.format(out_f))

if __name__ == "__main__":
    # usage: python3 extract_features.py -i INPUT_AUDIO_FILE -o OUTPUT_CSV_FILE
    # example: python3 extract_features.py -i ./test_audio/test.mp3 -o ./test_feats/test_mp3.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        help='Select input audio file [wav or mp3]',
                        action='store',
                        required=True,
                        dest='input')
    parser.add_argument('-o',
                        '--output',
                        help='Select output csv file',
                        action='store',
                        required=True,
                        dest='output')
    args = parser.parse_args()

    if os.path.exists(args.input) is False:
        print('Select existing input file!')
        sys.exit(0)
    if args.output.lower().endswith('.csv') is False:
        print('Output must be a csv file!')
        sys.exit(0)

    extract_features(args.input, args.output)
