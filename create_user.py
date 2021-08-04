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
from settings import *


def create_user(u_id):
    # create users folder
    user_path = os.path.join(path_models_users, str(u_id))
    try:
        os.makedirs(user_path)
    except FileExistsError:
        print('User has already been created!')
        sys.exit(0)

    # copy initial models
    pre_models = [os.path.join(root, f) for root, dirs, files in os.walk(path_models)
                  for f in files if (f.lower().endswith('.pkl') and 
                  os.path.join(root, f).count(os.path.sep) <= path_models.count(os.path.sep) + 1)]
    cp_models = [f.replace(path_models, user_path) for f in pre_models]

    for in_f, out_f in zip(pre_models, cp_models):
        shutil.copy(in_f, out_f)
    print('User {} created!'.format(u_id))


if __name__ == "__main__":
    # usage: python3 create_user.py -i USER_ID
    # example: python3 create_user.py -i 827
    # average time: Process lasted 0.899834394454956 seconds!
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_user',
                        help='Input user ID to create model and copy baseline classifiers',
                        action='store',
                        required=True,
                        dest='input_user')
    args = parser.parse_args()

    try:
        # TODO: user ids are created as numbers, will this change?
        user_id = int(args.input_user)
    except ValueError:
        print('User ID is invalid')
        sys.exit(0)

    create_user(user_id)
    print('Process lasted {} seconds!'.format((time.time()-start)))
