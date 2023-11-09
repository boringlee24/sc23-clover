import json
import socket 
import time
import requests
import math
import random

def urls_ok(url_list):
    for url in url_list:
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.01)
    return True

def get_accuracy(service_name, root_dir, response_time):
    with open(f'{root_dir}/{service_name}/service.json') as f:
        service = json.load(f)    
    with open(f'{root_dir}/{service_name}/accuracy.json') as f:
        accuracy = json.load(f)            
    num_req_sum = 0
    total_acc = 0
    for key, val in service.items():
        num_requests = len(response_time[val[1]])
        num_req_sum += num_requests
        acc = accuracy[val[0].split('_')[0]]
        total_acc += num_requests * acc
    return round(total_acc / num_req_sum, 2)

def parse_service_ip(service_name, root_dir):
    service_list = []        
    with open(f'{root_dir}/{service_name}/service.json') as f:
        load = json.load(f)
    for key, val in load.items():
        service_list.append(val[1])
    return service_list

def parse_tracker_ip(service_name, root_dir):
    tracker_list = []        
    with open(f'{root_dir}/{service_name}/tracker.json') as f:
        load = json.load(f)
    for key, val in load.items():
        tracker_list.append(val)
    return tracker_list

def send_signal(node, port=10000, cmd='test'):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = (node, int(port))

    print('connecting to {} port {}'.format(*server_address))
    sock.connect(server_address)

    try:
        # Send data
        message = cmd.encode('utf-8') #b'save 35'  #b'start 35 gpu 6'#b'save 35'
 
        print('sending {!r}'.format(message))
        sock.sendall(message)
        while True:
            data = sock.recv(32)
            if 'success' in data.decode('utf-8'):
                print('received {!r}'.format(data))
                break
            else:
                print('waiting for success signal')
                time.sleep(1)
    finally:
        #print('closing socket')
        sock.close()

def post_req(URL, ROUTE, request):    
    response = requests.post(f'{URL}/{ROUTE}', json=request)
    if not response.ok:
        raise RuntimeError(f'url {URL}/{ROUTE} not available')
    return response

def node_config_req(args, URL, logger):
    if args.mode == 'term':        
        req_json = {'term': ''}
    elif args.mode == 'multi_config': # config gpu
        req_json = {'multi_config': [args.num_gpu, args.mig_config]}
    elif args.mode == 'multi_start': # mix models on MIG device
        # the weights will be a dict, mapping each model to a mig slice
        # --weights xx6l means v5x, v5x, v5x6, v5l
        weight_list = []
        if args.service_name == 'yolo':
            for char in args.weights:
                if char == '6':
                    weight_list.append('yolov5x6')
                elif char == 'x':
                    weight_list.append('yolov5x')
                elif char == 'l':
                    weight_list.append('yolov5l')
                else:
                    raise RuntimeError('invalid char in weight')
        elif args.service_name == 'albert':
            models = ['base', 'large', 'xlarge', 'xxlarge']
            for char in args.weights:
                weight_list.append(models[int(char)])
        elif args.service_name == 'efficientnet':
            for char in args.weights:
                weight_list.append(f'b{char}')
        else:
            raise RuntimeError('invalid service name')
        req_json = {'multi_start': [args.num_gpu, args.mig_config], 
                    'weights': weight_list}
    else:
        raise RuntimeError('Error: No mode specified.')

    test_response = requests.post(f'{URL}/config', json=req_json)
    if not test_response.ok:
        raise RuntimeError('GPU node master service not available')
    else:
        logger.info('configured GPU node successfully')

class IP_port:
    def __init__(self, ip='127.0.0.0', port=0):
        self.ip = ip
        self.port = port

# class EvalResult:
#     def __init__(self, acc, lat, carbon):
#         self.acc = acc
#         self.lat = lat
#         self.carbon = carbon

class NodeConfigArgs:
    def __init__(self, mode, num_gpu, mig_config, weights, service_name) -> None:
        self.mode = mode        
        self.mig_config = mig_config
        self.weights = weights
        self.service_name = service_name
        self.num_gpu = num_gpu

class ObjectiveFunction:
    def __init__(self, Lambda, base_acc, base_carbon):
        self.Lambda = Lambda # between 0 and 1
        self.base_acc = base_acc
        self.base_carbon = base_carbon # per request average
    
    def effective_carbon_reduction(self, curr_acc, curr_energy, ci):
        # lambda*(deltaCarbon) - (1-lambda)*(deltaAcc)
        deltaCarbon = (self.base_carbon - curr_energy * ci) / self.base_carbon
        deltaAcc = (self.base_acc - curr_acc ) / self.base_acc
        objective = self.Lambda*(deltaCarbon) - (1-self.Lambda)*(deltaAcc)
        return objective, deltaAcc, deltaCarbon

    def effective_carbon_reduction_system(self, curr_acc, curr_carbon):
        # lambda*(deltaCarbon) - (1-lambda)*(deltaAcc)
        deltaCarbon = (self.base_carbon - curr_carbon) / self.base_carbon
        deltaAcc = (self.base_acc - curr_acc ) / self.base_acc
        objective = self.Lambda*(deltaCarbon) - (1-self.Lambda)*(deltaAcc)
        return objective, deltaAcc, deltaCarbon

class SimulatedAnneling:
    def __init__(self, init_temp=100, delta_temp=5, final_temp=10, center_score=-1000, K=1E-4) -> None:
        self.init_temp = init_temp
        self.curr_temp = init_temp
        self.delta_temp = delta_temp
        self.final_temp = final_temp
        self.center_score = center_score
        self.K = K

    def initialize(self, center_score):
        self.curr_temp = self.init_temp
        self.center_score = center_score

    def move_center(self, new_score, logger):
        diff = self.center_score - new_score # if diff < 0, center_score < new_score, move to new center
        move = False
        if diff <= 0:
            logger.info(f'temp {self.curr_temp}, center: {self.center_score}, new: {new_score}, move center 100%')
            self.center_score = new_score
            move = True
        else:
            prob = math.exp((0 - diff)/ (self.K * self.curr_temp))
            logger.info(f'temp {self.curr_temp}, center: {self.center_score}, new: {new_score}, move center {prob*100}%')
            if random.uniform(0,1) <= prob:
                self.center_score = new_score
                move = True
        self.curr_temp -= self.delta_temp
        if self.curr_temp < self.final_temp:
            self.curr_temp = self.final_temp
        if move:
            logger.info('moved center to new one')
        else:
            logger.info('did not move center')
        return move





