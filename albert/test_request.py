import requests
import argparse
import glob
import pdb
import random
from time import perf_counter
import json

test_url = "http://10.99.103.104:5000/detect"
print_req = True

with open('request/dev-v2.0.json') as f:
    corpus = json.load(f)

# now randomly select a context

book = random.choice(corpus['data'])
paragraph = random.choice(book['paragraphs'])
context = paragraph['context']
question = random.choice(paragraph['qas'])['question']

request = {'context': context, 'question': question}
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
