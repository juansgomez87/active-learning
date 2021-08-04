import requests
import time
import pdb
import argparse
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('-u_id',
                    '--user_id',
                    help='Input user_id to test post functions',
                    action='store',
                    required=True,
                    dest='u_id')
parser.add_argument('-it',
                    '--iterations',
                    help='Input number of iterations to test',
                    action='store',
                    type=int,
                    required=True,
                    dest='it')
parser.add_argument('-user',
                    '--user',
                    help='Input username',
                    action='store',
                    required=True,
                    dest='user')
parser.add_argument('-pass',
                    '--password',
                    help='Input password',
                    action='store',
                    required=True,
                    dest='pw')
args = parser.parse_args()

user_id = args.u_id
url = 'http://127.0.0.1:5000/api/v0.1/users/{}'.format(user_id)
q_class = ['Q1', 'Q2', 'Q3', 'Q4']

# testing user creation
print('Testing user creation with API!')
# print('json request:')
dt_te = {'method': 'create_user',
         'data': {}}
print(dt_te)

start = time.time()
x = requests.post(url, json=dt_te, auth=(args.user, args.pw))
print(x.text)
print('--- Process lasted {} seconds'.format(time.time() - start))

for i in range(args.it):
	time.sleep(2)
	# testing get hard tracks
	print('Testing get hard tracks with API!')
	dt_te = {'method': 'get_hard_tracks',
	         'data': {}}
	print(dt_te)

	start = time.time()
	x = requests.post(url, json=dt_te, auth=(args.user, args.pw))

	data = x.json()
	print('--- Process lasted {} seconds'.format(time.time() - start))

	anno_dict = {_:random.choice(q_class) for _ in data}

	time.sleep(2)
	# testing retrain model
	print('Testing retrain model with API!')
	dt_te = {'method': 'retrain_model',
	         'data': anno_dict}
	print(dt_te)

	start = time.time()
	x = requests.post(url, json=dt_te, auth=(args.user, args.pw))
	print(x.text)
	print('--- Process lasted {} seconds'.format(time.time() - start))


pdb.set_trace()