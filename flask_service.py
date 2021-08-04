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

from settings import *
import pdb

app = flask.Flask(__name__)
auth = HTTPBasicAuth()
app.config["DEBUG"] = True

users = {
    "juan": generate_password_hash("tide123"),
    "nicolas": generate_password_hash("tide123")
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
    # 127.0.0.1:5000/api/v0.1/users/827
    if request.method == 'GET':
        # check if user exists
        if user_id in list_users:
            return 'User {} exists!'.format(user_id)
        else:
            return 'User {} does not exist!'.format(user_id)

    if request.method == 'POST':
        data = request.get_json()
        if data['method'] == 'create_user':
            if user_id in list_users:
                return 'User {} exists, not creating path!'.format(user_id)
            else:
                subprocess.run(['python3',
                                'create_user.py',
                                '-i',
                                user_id])
                return 'User {} was created!'.format(user_id)

        elif data['method'] == 'get_hard_tracks':
            if user_id not in list_users:
                return 'User {} does not exist, create user first!'.format(user_id)
            else:
                lines = subprocess.run(['python3',
                                'get_hard_tracks.py',
                                '-i',
                                user_id,
                                '-q',
                                '10'], stdout=subprocess.PIPE).stdout.splitlines()
                q_list = eval(lines[-1])

                return jsonify(q_list)

        elif data['method'] == 'retrain_model':
            if user_id not in list_users:
                return 'User {} does not exist, create user first!'.format(user_id)
            else:
                anno = data['data']
                path_user = os.path.join(path_models_users, user_id)
                json_fn = path_user + '/last_anno.json'
                with open(json_fn, 'w') as f:
                    json.dump(anno, f, indent=4)

                lines = subprocess.run(['python3',
                                'retrain_model.py',
                                '-i',
                                user_id,
                                '-a',
                                json_fn], stdout=subprocess.PIPE).stdout.splitlines()

                return 'Finished retraining models for user {}'.format(user_id)




if __name__ == "__main__":


    app.run()