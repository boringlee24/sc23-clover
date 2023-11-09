import requests
import argparse
import glob
import pdb
import random
# from threading import Thread
from multiprocessing import Process, Lock, Manager, Value
import time
import json
import os
import sys
from time import perf_counter
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
os.chdir(str(ROOT))

# keep updating values to 
def thread_task(lock, identifier, url, shared_dict, counter, model, args):
    #TODO: change it to sending question and context
    with open('request/dev-v2.0.json') as f:
        corpus = json.load(f)

    shared_list = []
    while True:
        book = random.choice(corpus['data'])
        paragraph = random.choice(book['paragraphs'])
        context = paragraph['context']
        question = random.choice(paragraph['qas'])['question']

        request = {'context': context, 'question': question}

        t0 = perf_counter()
        test_response = requests.post(url, json=request)
        t1 = perf_counter()
        lat_ms = round((t1-t0)*1000, 2)
        if not test_response.ok:
            raise RuntimeError(f'thread {identifier} did not receive proper response')
        shared_list.append(lat_ms)
        # write to shared memory
        with lock:
            # Global variables are not shared between processes.
            counter.value += 1
        if counter.value >= args.inf_num:
            shared_dict[model] = shared_list
            print(f'thread {identifier} finished')            
            break        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--inf_num', type=int, default=100, help='number of inferences')
    parser.add_argument('--testcase', type=str, default='p6_12', help='server configuration')
    args = parser.parse_args()

    with open('service.json') as f:
        service = json.load(f)
    with open('tracker.json') as f:
        tracker = json.load(f)
    Path('logs/perf/mixed_model').mkdir(parents=True, exist_ok=True)
    Path('logs/carbon/mixed_model').mkdir(parents=True, exist_ok=True)
    
    req_json = {
        'info': 'start',
        'testcase': f'mixed_model/result_b1_{args.testcase}'
    }
    tracker_url = list(tracker.values())[0]
    test_response = requests.post(f'{tracker_url}/track', json=req_json)
    if not test_response.ok:
        raise RuntimeError('Tracker service not available')
    
    with Manager() as manager:
        counter = Value('i', 0)
        shared_dict = manager.dict()
        processes = []
        lock = Lock()

        for i, (model, url) in enumerate(list(service.values())):
            p = Process(target=thread_task, args=(lock, i, f'{url}/detect', shared_dict, counter, model, args))
            print(f'process {i} requesting url {url}')
            processes.append(p)
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        with open(f'logs/perf/mixed_model/result_b1_{args.testcase}.json', 'w') as f:
            json.dump(dict(shared_dict), f, indent=4)

    req_json = {
        'info': 'end'
    }
    tracker_url = list(tracker.values())[0]
    test_response = requests.post(f'{tracker_url}/track', json=req_json)
    if not test_response.ok:
        raise RuntimeError('Tracker service not responding')

        


    

