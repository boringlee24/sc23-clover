import requests
import argparse
import glob
import pdb
import random
from time import perf_counter

test_url = "http://10.99.103.100:5100/carbon"
print_req = True

parser = argparse.ArgumentParser()
parser.add_argument('--ci', type=float, default=250, help='carbon intensity')
parser.add_argument('--pue', type=float, default=1.5, help='datacenter PUE')
args = parser.parse_args()        

request = {'ci': args.ci, 'pue': args.pue}
print(request)
# test_files = {}

send = perf_counter()
test_response = requests.post(test_url, json=request)
duration = round((perf_counter() - send)*1000, 3)
print(f'End-to-end response time: {duration}ms')

if print_req:
    header = test_response.request.headers
    body = test_response.request.body

if test_response.ok:
    print("Upload completed successfully!")
    print(test_response.text)
else:
    print("Something went wrong!")