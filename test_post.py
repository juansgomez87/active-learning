import requests
import time
import pdb
import argparse
import random
import json
import os
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('-u_id',
                    '--user_id',
                    help='Input user_id to test post functions',
                    action='store',
                    type=int,
                    required=True,
                    dest='u_id')
parser.add_argument('-it',
                    '--iterations',
                    help='Input number of iterations to test',
                    action='store',
                    type=int,
                    required=True,
                    dest='it')
parser.add_argument('-mode',
                    '--mode',
                    help='Input mode [mc, hc, mix, rand]',
                    action='store',
                    required=True,
                    dest='mode')
args = parser.parse_args()

user_id = args.u_id
# # no docker
# url = 'http://192.168.1.134:5000/api/v0.1/users/{}'.format(user_id)
# # no docker office
# url = 'http://10.80.25.42:5000/api/v0.1/users/{}'.format(user_id)
# # with docker
# url = 'http://172.17.0.2:5000/api/v0.1/users/{}'.format(user_id)
# # with docker-compose
# url = 'http://localhost:5000/api/v0.1/users/{}'.format(user_id)
# # with vpn
# url = 'http://mirlab-web1.s.upf.edu/active-learning/api/v0.1/users/{}'.format(user_id)
# # no vpn
url = 'https://trompa-mtg.upf.edu/active-learning/api/v0.1/users/{}'.format(user_id)
q_class = ['Q1', 'Q2', 'Q3', 'Q4']
recs_list = ['latin', 'africa', 'mideast']

# testing user creation
print('Testing user creation with API!')
# print('json request:')
dt_te = {'method': 'create_user',
         'data': args.mode}
print(dt_te)

start = time.time()
x = requests.post(url, json=dt_te, auth=(os.environ['USER_API'], os.environ['PASS_API']))
print('Returns:\n{}'.format(x.text))
print('--- Process lasted {} seconds'.format(time.time() - start))

for i in range(args.it):
	time.sleep(2)
	# testing get hard tracks
	print('Testing get hard tracks with API!')
	dt_te = {'method': 'get_hard_tracks',
	         'data': {}}
	print(dt_te)

	start = time.time()
	x = requests.post(url, json=dt_te, auth=(os.environ['USER_API'], os.environ['PASS_API']))
	data = x.json()
	print('Returns:\n{}'.format(x.text))
	print('--- Process lasted {} seconds'.format(time.time() - start))

	anno_dict = {_:random.choice(q_class) for _ in data}

	time.sleep(2)
	# testing retrain model
	print('Testing retrain model with API!')
	dt_te = {'method': 'retrain_model',
	         'data': anno_dict}
	print(dt_te)

	start = time.time()
	x = requests.post(url, json=dt_te, auth=(os.environ['USER_API'], os.environ['PASS_API']))
	print('Returns:\n{}'.format(x.text))
	print('--- Process lasted {} seconds'.format(time.time() - start))


	rec = random.choice(recs_list)
	dt_te = {'method': 'get_recommendations',
	         'data': rec}
	print(dt_te)
	x = requests.post(url, json=dt_te, auth=(os.environ['USER_API'], os.environ['PASS_API']))
	print('Returns:\n{}'.format(x.text))
	print('--- Process lasted {} seconds'.format(time.time() - start))

pdb.set_trace()

start = time.time()
dt_te = {'method': 'delete_user',
         'data': args.mode}
print(dt_te)
x = requests.post(url, json=dt_te, auth=(os.environ['USER_API'], os.environ['PASS_API']))
print('Returns:\n{}'.format(x.text))
print('--- Process lasted {} seconds'.format(time.time() - start))
