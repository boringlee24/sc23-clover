import requests
import argparse
import glob
import pdb
import random
from time import perf_counter
import json

test_url = "http://10.99.103.101:5200/config"
# request = {'multi_start': [2, '10'], 'weights': ['yolov5l', 'yolov5x', 'yolov5x6', 'yolov5x6', 'yolov5x']}
request = {'term': ''}
# request = {'multi_config': [2, '10']}
print(request)
# test_files = {}

send = perf_counter()
test_response = requests.post(test_url, json=request)
duration = round((perf_counter() - send)*1000, 3)
print(f'End-to-end response time: {duration}ms')

if test_response.ok:
    print("Upload completed successfully!")
    print(test_response.text)
else:
    print("Something went wrong!")
