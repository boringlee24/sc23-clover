import argparse
import json
import signal
import time 
import threading
from time import perf_counter
import numpy as np
from collections import deque
from multiprocessing import Event, Queue, Process
from requests_futures.sessions import FuturesSession
import random
import glob
from system_utils import parse_service_ip, send_signal, get_accuracy
from pathlib import Path
import socket
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
REPOROOT=FILE.parents[2]
import logging.config
logging.config.fileConfig(fname=f'{ROOT}/logging_gen.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

def producer(stop_event, rate, queue):
    np.random.seed(1)
    while not stop_event.is_set():
        queue.put(perf_counter())
        time.sleep(np.random.exponential(rate))        
    logger.info('stopped request generating thread')

def consumer(stop_event, input_list, queue, gen_session, controller_ip, limit):

    def response_hook(response, *request_args, **request_kwargs):
        # retrieve the ip that the response comes from
        source_ip = response.url.strip('/detect')

        start_time = gen_session.req_record[source_ip]
        gen_session.ip_state.appendleft(source_ip)
        gen_session.response_time[source_ip].append(round((perf_counter() - start_time)*1000, 2)) # ms unit
        gen_session.req_cnt += 1
    
    while gen_session.req_cnt <= limit:
        if gen_session.service in ['yolo', 'efficientnet']:
            if not queue.empty() and len(gen_session.ip_state) > 0:
                # generate async request
                url = gen_session.ip_state.pop()
                q_time = queue.get()
                gen_session.req_record[url] = q_time
                request = random.sample(input_list, 8)
                test_files = []        
                for i in range(len(request)):
                    test_files.append(('img', open(request[i], 'rb')))           
                gen_session.session.post(f'{url}/detect', files=test_files, hooks={'response': response_hook})
        elif gen_session.service == 'albert':
            if not queue.empty() and len(gen_session.ip_state) > 0:
                # generate async request
                url = gen_session.ip_state.pop()
                q_time = queue.get()
                gen_session.req_record[url] = q_time
                book = random.choice(input_list['data'])
                paragraph = random.choice(book['paragraphs'])
                context = paragraph['context']
                question = random.choice(paragraph['qas'])['question']
                request = {'context': context, 'question': question}
                gen_session.session.post(f'{url}/detect', json=request, hooks={'response': response_hook})            
    # send_signal(controller_ip['ip'], controller_ip['port'], cmd=f'stop tracker')
    logger.info('stopped request consumer')            

def get_accuracy_and_latency(service_name, response_time):    
    avg_acc = get_accuracy(service_name=service_name, root_dir=FILE.parents[1], response_time=response_time)
    # p95 tail latency
    all_lat = []
    for k, v in response_time.items():
        all_lat += v
    tail_lat = round(np.percentile(all_lat, 95), 2)
    return avg_acc, tail_lat

def tcp_listener(controller_ip): # controller thread listen to GPU node feedback
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (socket.gethostbyname(socket.gethostname()), 10002)
    logger.info('starting up on {} port {}'.format(*server_address))
    # print('starting up on {} port {}'.format(*server_address), file=run_log, flush=True)
    sock.bind(server_address)
    sock.listen(5) 

    while True:
        # Wait for a connection
        connection, client_address = sock.accept()      
        try:
            while True:
                data = connection.recv(32)
                if data: 
                    data_str = data.decode('utf-8')
                    if 'start' in data_str: # start yolo 2000 0.1
                        # start the producer and consumer processes
                        data_str = data_str.split(' ')
                        service_name, limit, arrival = data_str[1], int(data_str[2]), float(data_str[3])                        
                        service_ip = parse_service_ip(service_name, FILE.parents[1])
                        queue = Queue() # queues don't need locks
                        stop_event = Event()
                        gen_session = GeneratorSession(service_ip=service_ip, service=service_name)
                        if service_name == 'yolo':
                            input_list = glob.glob(f'{REPOROOT}/datasets/coco/images/val2017/*.jpg')
                        elif service_name == 'efficientnet':
                            input_list = glob.glob(f'{REPOROOT}/efficient-net/examples/imagenet/data/val/*/*.JPEG')
                        elif service_name == 'albert':
                            with open(f'{REPOROOT}/carbon_scheduler/albert/request/dev-v2.0.json') as f:
                                input_list = json.load(f)                                                        
                        else:
                            raise RuntimeError('Service is not supported')

                        p1 = Process(target=producer, args=(stop_event,arrival,queue))      
                        p1.start()
                        # p2 = Process(target=consumer, args=(stop_event,input_list,args,queue,gen_session))
                        # p2.start()
                        # p2.join()
                        consumer(stop_event, input_list=input_list, queue=queue, gen_session=gen_session, controller_ip=controller_ip, limit=limit)
                        stop_event.set()
                        p1.join()
                        # logger.info(gen_session.response_time)
                        accuracy, latency = get_accuracy_and_latency(service_name, gen_session.response_time)
                        send_signal(controller_ip['ip'], controller_ip['port'], cmd=f'perf {accuracy} {latency}')
                        # connection.close()
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()

class GeneratorSession:
    def __init__(self, service_ip, service):
        self.session = FuturesSession()
        self.ip_state = deque(service_ip)
        self.req_record = {}
        self.response_time = {}
        for ip in service_ip:
            self.req_record[ip] = ''
            self.response_time[ip] = []
        self.req_cnt = 0        
        self.service = service
    # def new_session(self, service_ip):
    #     self.session = FuturesSession()
    #     self.ip_state = deque(service_ip)
    #     self.req_record = {}
    #     self.response_time = {}
    #     for ip in service_ip:
    #         self.req_record[ip] = ''
    #         self.response_time[ip] = []
    #     self.req_cnt = 0        

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='service details')
    # parser.add_argument('--arrival', type=float, help='inter-arrival period', default=0.2)
    # parser.add_argument('--service', type=str, help='type of service (e.g., albert)', default='yolo')
    # parser.add_argument('--batch', type=int, help='request batch size', default=8)
    # # parser.add_argument('--limit', type=int, help='request number limit before termination', default=200)    
    # args = parser.parse_args()

    with open('controller.json') as f:
        controller_ip = json.load(f)
    hostname = socket.gethostbyname(socket.gethostname())
    send_signal(controller_ip['ip'], controller_ip['port'], cmd=f'gen ip {hostname}')

    tcp_listener(controller_ip)