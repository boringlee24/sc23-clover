import requests
import argparse
import glob
import pdb
import random
# from threading import Thread
from multiprocessing import Process, Lock, Manager
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
def thread_task(lock, identifier, url, shared_list, args):
    images = glob.glob('/work/li.baol/GIT/datasets/coco/images/val2017/*.jpg')

    while True:
        request = random.sample(images, args.batch)

        test_files = []        
        for i in range(len(request)):
            test_files.append(('img', open(request[i], 'rb')))   

        t0 = perf_counter()
        test_response = requests.post(url, files=test_files)
        t1 = perf_counter()
        lat_ms = round((t1-t0)*1000, 2)
        if not test_response.ok:
            raise RuntimeError(f'thread {identifier} did not receive proper response')
        # write to common list
        with lock:
            # Global variables are not shared between processes.
            shared_list.append(lat_ms)
            if len(shared_list) >= args.inf_num:
                print(f'thread {identifier} finished')
                break        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='inference batch size')
    parser.add_argument('--inf_num', type=int, default=100, help='number of inferences')
    parser.add_argument('--testcase', type=str, default='p6_x6', help='server configuration')
    args = parser.parse_args()

    with open('service.json') as f:
        service = json.load(f)
    with open('tracker.json') as f:
        tracker = json.load(f)
    req_json = {
        'info': 'start',
        'testcase': f'single_model/result_b{args.batch}_{args.testcase}'
    }
    tracker_url = list(tracker.values())[0]
    test_response = requests.post(f'{tracker_url}/track', json=req_json)
    if not test_response.ok:
        raise RuntimeError('Tracker service not available')

    with Manager() as manager:
        shared_list = manager.list()
        processes = []
        lock = Lock()

        for i, url in enumerate(list(service.values())):
            p = Process(target=thread_task, args=(lock, i, f'{url}/detect', shared_list, args))
            print(f'process {i} requesting url {url}')
            processes.append(p)
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        with open(f'logs/perf/single_model/result_b{args.batch}_{args.testcase}.json', 'w') as f:
            json.dump(list(shared_list), f, indent=4)
    req_json = {
        'info': 'end'
    }
    tracker_url = list(tracker.values())[0]
    test_response = requests.post(f'{tracker_url}/track', json=req_json)
    if not test_response.ok:
        raise RuntimeError('Tracker service not available')

        


    

