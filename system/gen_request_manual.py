import argparse
import json
import signal
import time 
import threading
from collections import deque
from time import perf_counter
import numpy as np
from threading import Event
from requests_futures.sessions import FuturesSession
import random
import glob
from system_utils import parse_service_ip
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
import logging.config
logging.config.fileConfig(fname=f'{ROOT}/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='service details')
parser.add_argument('--arrival', type=float, help='inter-arrival period', default=0.2)
parser.add_argument('--service', type=str, help='type of service (e.g., albert)', default='yolo')
parser.add_argument('--batch', type=int, help='request batch size', default=8)
args = parser.parse_args()

# with open('service.json') as f:
#     service_json = json.load(f)
# service_ip = service_json[args.service]
service_ip = parse_service_ip(args.service, FILE.parents[1])

req_record = {} # track which request is sent to the IP address by its unique perf_counter()
response_time = {} # round-trip response time of each request
for ip in service_ip:
    req_record[ip] = ''
    response_time[ip] = []

ip_state = deque(service_ip)
start_proc = False
session = FuturesSession()
stop_producer = Event()
stop_consumer = Event()

if args.service == 'yolo':
    input_list = glob.glob('/work/li.baol/GIT/datasets/coco/images/val2017/*.jpg')

# wait for starting signal
def startOrStopProcess(signalNumber, frame):
    global start_proc, stop_consumer, response_time
    if any(v for v in response_time.values()): # stop producer and consumer
        if not stop_producer.is_set():
            stop_producer.set()
        else:
            stop_consumer.set()
    else:
        start_proc = True

def producer(stop_event, rate):
    time.sleep(1) # wait for consumer to come online
    global q
    while not stop_event.is_set():
        q.appendleft(perf_counter())
        time.sleep(np.random.exponential(rate))
    logger.info('stopped request generating thread')

def response_hook(response, *request_args, **request_kwargs):
    global req_record, ip_state, response_time
    # retrieve the ip that the response comes from
    source_ip = response.url.strip('/detect')

    start_time = req_record[source_ip]
    ip_state.appendleft(source_ip)
    response_time[source_ip].append(round((perf_counter() - start_time)*1000, 2)) # ms unit

def consumer(stop_event, input_list):
    global q, req_record, ip_state
    while not stop_event.is_set():
        if len(q) > 0 and len(ip_state) > 0:
            # generate async request
            url = ip_state.pop()
            q_time = q.pop()
            req_record[url] = q_time
            request = random.sample(input_list, args.batch)
            test_files = []        
            for i in range(len(request)):
                test_files.append(('img', open(request[i], 'rb')))   

            session.post(f'{url}/detect', files=test_files, hooks={'response': response_hook})
    logger.info('stopped request consumer')            

signal.signal(signal.SIGINT, startOrStopProcess)

while not start_proc:
    time.sleep(0.1)

logger.info(f'started querying {args.service} service')

# thread to generate request and record arrival time 

q = deque()

x = threading.Thread(target=producer, daemon=True, args=(stop_producer, args.arrival))
x.start()
consumer(stop_consumer, input_list)

# time.sleep(3)


pass
