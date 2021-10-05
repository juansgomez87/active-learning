#!/usr/bin/env python3
"""
Settings


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import argparse
import pandas as pd
import sys
import flask
from flask import request
from flask import jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import glob
import subprocess
import os
import json
from dotenv import load_dotenv

from settings import *
from create_user import create_user
from get_hard_tracks import ConsensusEntropyCalculator
from retrain_model import Retrainer

load_dotenv()

app = flask.Flask(__name__)
auth = HTTPBasicAuth()
app.config["DEBUG"] = True

users = {
    os.environ['USER_API']: generate_password_hash(os.environ['PASS_API']),
}

@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username

@app.route('/', methods=['GET'])
@auth.login_required
def home():
    return "<h1>TROMPA Music Emotion Recognition</h1><p>This site is a prototype API for the ME pilot.</p>"

@app.route('/api/v0.1/users/<user_id>', methods=['GET', 'POST'])
@auth.login_required
def user(user_id):
    list_users = [_.split('/')[3] for _ in glob.glob(path_models_users+ '/*/')]
    path_user = os.path.join(path_models_users, user_id)
    if request.method == 'GET':
        # check if user exists
        if user_id in list_users:
            return 'User {} exists!'.format(user_id)
        else:
            return 'User {} does not exist!'.format(user_id)

    if request.method == 'POST':
        data = request.get_json()
        if data['method'] == 'create_user':
            mode = data['data']
            if user_id in list_users:
                return 'User {} exists, not creating path!'.format(user_id)
            else:
                create_user(user_id, mode)

                return 'User {} was created with mode {}!'.format(user_id, mode)

        elif data['method'] == 'get_hard_tracks':
            if user_id not in list_users:
                return 'User {} does not exist, create user first!'.format(user_id)
            else:
                mode = [d for root, dirs, files in os.walk(os.path.join(path_models_users, user_id)) for d in dirs][0]
                q_list = ConsensusEntropyCalculator(10, path_user).run()

                return jsonify(q_list)

        elif data['method'] == 'retrain_model':
            if user_id not in list_users:
                return 'User {} does not exist, create user first!'.format(user_id)
            else:
                anno = data['data']
                recs = data['recs']
                
                mode = [d for root, dirs, files in os.walk(os.path.join(path_models_users, user_id)) for d in dirs][0]
                json_fn = os.path.join(path_user, mode, 'last_anno.json')
                with open(json_fn, 'w') as f:
                    json.dump(anno, f, indent=4)
                rec_list = Retrainer(json_fn, path_user, recs).run()

                return jsonify(rec_list)

        elif data['method'] == 'delete_user':
            if user_id in list_users:
                subprocess.run(['rm', '-r', path_user])
                return 'User {} deleted!'.format(user_id)

            else:
                return 'Send correct user!'


if __name__ == "__main__":
    app.run(host='0.0.0.0')