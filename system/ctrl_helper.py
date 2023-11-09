import time
import socket
import sys
from system_utils import parse_service_ip, parse_tracker_ip, send_signal
import system_utils
import requests
import json

def interpret_ctrl(data_str, runtime, logger):
    # 
    if 'gen ip' in data_str: # gen ip 123456
        data_str = data_str.split(' ')
        runtime.generator_ip.ip = data_str[2]
        runtime.generator_ip.port = 10002
        logger.info(f'received generator information from {runtime.generator_ip.ip}')
    elif 'gpu_node ip' in data_str: # gpu_node ip 123456
        runtime.gpu_node_ip.append(data_str.split(' ')[2])
    elif 'perf' in data_str: # perf 90.5 123.45
        data_str = data_str.split(' ')
        acc = float(data_str[1])
        lat = float(data_str[2])
        # record acc, latency, then send request to stop carbon tracker
        tracker_ips = parse_tracker_ip(runtime.service_name, runtime.root_dir)
        req_json = {
            'info': 'end_return'
        }
        carbon = 0
        for tracker_ip in tracker_ips:
            test_response = system_utils.post_req(tracker_ip, 'track', request=req_json)
            response_json = json.loads(test_response.content)
            logger.info(response_json) 
            carbon += response_json['co2']
        carbon = carbon / runtime.req_limit # per-request carbon
        runtime.eval_result['acc'].append(acc)
        runtime.eval_result['lat'].append(lat)
        runtime.eval_result['carbon'].append(carbon)
    elif 'term_thread' in data_str:                        
        pass
    else:
        raise RuntimeError('message to controller cannot be decoded')

def ctrl_thread(runtime, logger): # controller thread listen to GPU node feedback
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (socket.gethostbyname(socket.gethostname()), 10002)
    logger.info('starting up on {} port {}'.format(*server_address))
    # print('starting up on {} port {}'.format(*server_address), file=run_log, flush=True)
    sock.bind(server_address)
    sock.listen(5) 

    while not runtime.stop_event.is_set():
        # Wait for a connection
        connection, client_address = sock.accept() # this is a blocking statement
        try:
            while True:
                data = connection.recv(32)
                if data: 
                    data_str = data.decode('utf-8')
                    interpret_ctrl(data_str, runtime, logger)
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()
            # print('Terminated connection')
    print('Terminated thread')
