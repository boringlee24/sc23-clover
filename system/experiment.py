import socket
from threading import Event
from system_utils import send_signal, IP_port, NodeConfigArgs, node_config_req, urls_ok
import system_utils
import threading
from ctrl_helper import ctrl_thread
from pathlib import Path
import time
import json
import pandas as pd
import numpy as np
import random
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
GITROOT=FILE.parents[1]
import logging.config
logging.config.fileConfig(fname=f'{ROOT}/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class BaseExperiment:
    def __init__(self, service_name, num_nodes=1, gpus_per_node=1):
        self.service_name = service_name
        self.root_dir = str(GITROOT)
        self.stop_event = Event() # stop the background thread
        self.generator_ip = IP_port()
        self.gpu_node_ip = [] # hostname/ip addr of the GPU nodes
        self.iteration = 0
        self.iter_limit = 1
        self.ci_profile = [200, 300]
        self.pue_profile = [1.5, 1.5]
        self.eval_result = {} #{'algo': {'lat':[], 'acc': [], 'carbon': []}}
        self.exp_algorithm = 'base'
        self.req_limit = 2000 # 1000 requests to evaluate the configuration
        self.listener_thread = ''
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.arrival = 0.5
        self.obj_func = system_utils.ObjectiveFunction(0,0,0)
        self.duplicated_keys = ['9', '15', '16', '17']
        self.mig_partition = ''
        self.models = ''
        self.avg_ci = np.mean(self.ci_profile)
        self.avg_pue = np.mean(self.pue_profile)
        self.base_unit_carbon = 0 # using baseline, carbon per request with ci=1, pue=1

    def setup_obj_func(self, Lambda, ci_base, pue_base):
        with open(f'{self.root_dir}/system/logs/base/{self.service_name}.json') as f:
            baseline = json.load(f)
        base_acc = baseline['acc'][0]
        self.avg_ci = ci_base
        self.avg_pue = pue_base
        self.base_unit_carbon = baseline['unit_carbon']
        base_carbon = baseline['unit_carbon'] * self.avg_ci * self.avg_pue
        self.obj_func = system_utils.ObjectiveFunction(Lambda=Lambda,base_acc=base_acc,base_carbon=base_carbon)

    def get_ci_and_pue(self, ci_path='', pue_path=''):
        df = pd.read_csv(ci_path)
        self.ci_profile = df['carbon_intensity_production_avg'].to_list()
        if pue_path == '':
            self.pue_profile = [1.5] * len(self.ci_profile)
        else:
            df = pd.read_csv(pue_path)
            self.pue_profile = df['pue'].to_list()

    def term_thread(self):
        # KILL THE THREAD
        message = f'term_thread'
        self.stop_event.set()
        system_utils.send_signal(socket.gethostbyname(socket.gethostname()), 10002, message)

    def reconfig_mig(self, mig_partition, models, reset_mig=True):
        node_config = NodeConfigArgs(mode='config', num_gpu=self.gpus_per_node, 
                mig_config=mig_partition, weights=models, service_name=self.service_name)
        for node in self.gpu_node_ip:
            node_config.mode = 'term'
            node_config_req(node_config, f'http://{node}:5200', logger=logger)
            time.sleep(0.5)

            if reset_mig:
                time.sleep(0.5)
                node_config.mode = 'multi_config'
                node_config_req(node_config, f'http://{node}:5200', logger=logger)
                time.sleep(0.5) # give a bit slack to avoid crashing the system
            node_config.mode = 'multi_start' # this starts both inference service and tracker service
            node_config_req(node_config, f'http://{node}:5200', logger=logger)
        
        service_ips = system_utils.parse_service_ip(self.service_name, self.root_dir)
        tracker_ips = system_utils.parse_tracker_ip(self.service_name, self.root_dir)
        logger.info('Waiting for service URLs to be up')
        time.sleep(1)
        urls_ok(service_ips + tracker_ips)
        logger.info('All inference services ready')

    def initialize(self, mig_partition='0', models='x', wait=True, gpu_node_ip='', generator_ip='', generator_port=''):
        self.mig_partition = mig_partition
        self.models = models
        self.listener_thread = threading.Thread(target=ctrl_thread, daemon=True, args=(self, logger))
        self.listener_thread.start()
        hostname = socket.gethostname()
        controller_ip = {'ip': socket.gethostbyname(hostname), 'port': 10002}
        with open('controller.json', 'w') as f:
            json.dump(controller_ip, f, indent=4)
        
        if wait:
            logger.info('wait for GPU node server to be up')
            while True:
                if len(self.gpu_node_ip) == self.num_nodes:
                    break
                time.sleep(0.1)
            time.sleep(1)
            logger.info('GPU node server is up, configuring GPU nodes')
        else:
            self.gpu_node_ip = gpu_node_ip
        # initialize the GPU nodes
        self.reconfig_mig(mig_partition=mig_partition, models=models, reset_mig=True)

        if wait:
            logger.info('Wait for the generator node to be up')
            while True:
                if self.generator_ip.port != 0:
                    break
                time.sleep(0.1)
            time.sleep(1)
        else:
            self.generator_ip.ip, self.generator_ip.port = generator_ip, generator_port
        logger.info('Initialization done!')
        return self.gpu_node_ip, self.generator_ip.ip, self.generator_ip.port

    def run(self):        
        # make logging directory if not exist
        Path(f'{self.root_dir}/system/logs/{self.exp_algorithm}').mkdir(parents=True, exist_ok=True)        
        self.eval_result = {'acc': [], 'lat': [], 'carbon': []}
        tracker_ips = system_utils.parse_tracker_ip(self.service_name, self.root_dir)
        self.get_ci_and_pue(ci_path='csv/carbon_intensity.csv')
        if self.service_name == 'albert':
            self.arrival = 0.4
        elif self.service_name in ['yolo', 'efficientnet']:
            self.arrival = 1 # equivalent to 0.5 rate on 4 GPUs

        req_json = {'ci': 1, 'pue': 1}
        for tracker_ip in tracker_ips:
            system_utils.post_req(tracker_ip, 'carbon', request=req_json)
        # start tracker
        req_json = {'info': 'start', 'testcase': ''}
        for tracker_ip in tracker_ips:
            system_utils.post_req(tracker_ip, 'track', request=req_json)  
        
        logger.info(f'started evaluation of baseline')
        # start request generator, BLOCKING
        send_signal(self.generator_ip.ip, self.generator_ip.port, 
                    cmd=f'start {self.service_name} {self.req_limit} {self.arrival}')
        logger.info(f'finished evaluation of baseline')
        # extrapolate and save to json
        self.eval_result['unit_carbon'] = self.eval_result['carbon'][0]
        self.eval_result['acc'] = self.eval_result['acc'] * len(self.ci_profile)
        self.eval_result['carbon'] = np.multiply(self.eval_result['carbon'][0] * np.array(self.ci_profile), np.array(self.pue_profile)).tolist()
        self.eval_result['lat'] = self.eval_result['lat'] * len(self.ci_profile)
        with open(f'{self.root_dir}/system/logs/{self.exp_algorithm}/{self.service_name}.json', 'w') as f:
            json.dump(self.eval_result, f, indent=4)
        self.term_thread()
        self.listener_thread.join()

    def sweep_arrival(self, algo):
        self.exp_algorithm = algo
        Path(f'{self.root_dir}/system/logs/arrival/{self.exp_algorithm}/{self.service_name}').mkdir(parents=True, exist_ok=True)        
        self.eval_result = {'acc': [], 'lat': [], 'carbon': []}
        tracker_ips = system_utils.parse_tracker_ip(self.service_name, self.root_dir)

        if self.service_name == 'albert':
            arrival_list = [0.4, 0.16, 0.08]
        elif self.service_name in ['yolo', 'efficientnet']:
            arrival_list = [1, 0.4, 0.2]
        self.seed = 0

        for i, arrival in enumerate(arrival_list):
            self.arrival = arrival
            req_json = {'ci': 1, 'pue': 1}
            for tracker_ip in tracker_ips:
                system_utils.post_req(tracker_ip, 'carbon', request=req_json)
            # start tracker
            req_json = {'info': 'start', 'testcase': ''}
            for tracker_ip in tracker_ips:
                system_utils.post_req(tracker_ip, 'track', request=req_json)              

            logger.info(f'started evaluation of {arrival}')
            # start request generator, BLOCKING
            self.req_limit = 2000 if i == 0 else 1000
            send_signal(self.generator_ip.ip, self.generator_ip.port, 
                        cmd=f'start {self.service_name} {self.req_limit} {self.arrival}')
            logger.info(f'finished evaluation of {arrival}')
            with open(f'{self.root_dir}/system/logs/arrival/{self.exp_algorithm}/{self.service_name}/arrival={arrival}.json', 'w') as f:
                json.dump(self.eval_result, f, indent=4)
            time.sleep(10)
        self.term_thread()
        self.listener_thread.join()

    def test_run(self):        
        self.eval_result = {'acc': [], 'lat': [], 'carbon': []}
        tracker_ips = system_utils.parse_tracker_ip(self.service_name, self.root_dir)
        self.req_limit = 500
        if self.service_name == 'albert':
            self.arrival = 0.4
        elif self.service_name in ['yolo', 'efficientnet']:
            self.arrival = 1
        # START tracker
        for i in range(self.iter_limit): # exploration iterations
            req_json = {'ci': self.ci_profile[i], 'pue': self.pue_profile[i]}
            for tracker_ip in tracker_ips:
                system_utils.post_req(tracker_ip, 'carbon', request=req_json)
            # start tracker
            req_json = {'info': 'start', 'testcase': ''}
            for tracker_ip in tracker_ips:
                system_utils.post_req(tracker_ip, 'track', request=req_json)  

            logger.info(f'started evaluation of iteration {i}')
            # start request generator, BLOCKING
            send_signal(self.generator_ip.ip, self.generator_ip.port, 
                        cmd=f'start {self.service_name} {self.req_limit} {self.arrival}')
            logger.info(f'finished evaluation of iteration {i}')
            # wait for generator and tracker number to be available
            time.sleep(1)

        self.term_thread()
        self.listener_thread.join()
        self.stop_event.clear()
    
class RandomExperiment(BaseExperiment): # with history, but no MIG graph
    def __init__(self, service_name, num_nodes=1, gpus_per_node=1):
        super().__init__(service_name, num_nodes, gpus_per_node)
        with open(f'{self.root_dir}/system/logs/{self.exp_algorithm}/{self.service_name}.json') as f:
            read = json.load(f)
        self.SLA = read['lat'][0]
        self.exp_algorithm = 'rand'
        if self.service_name == 'albert':
            self.arrival = 0.08
        elif self.service_name in ['yolo', 'efficientnet']:        
            self.arrival = 0.2
        self.seed = 0
        self.eval_result_seq = []
        self.evaluated_config = {}
        self.evaluated_config_set = set()
        self.iter_limit = 20 # if after 5 iterations no improvement, stop
        self.no_improv_limit = 5
        self.req_limit = 1000
        with open(f'{self.root_dir}/mig/partition_code.json') as f:
            self.partition_code = json.load(f)
        self.possible_partitions = [k for k in self.partition_code if k not in self.duplicated_keys]
        self.possible_models = {'yolo': ['l', 'x', '6'], 
                                'albert': ['0', '1', '2', '3'], 
                                'efficientnet': ['1', '3', '5', '7']}                

    def gen_random_mig(self, curr_hour):
        mig_partition = random.choice(self.possible_partitions)
        models = ''        
        for i in range(len(self.partition_code[mig_partition])):
            models += random.choice(self.possible_models[self.service_name])          
        # check if already seen in current iter
        if (mig_partition, models) in self.evaluated_config[curr_hour]:
            mig_partition, models = self.gen_random_mig(curr_hour)
        return mig_partition, models      
        # self.mig_partition, self.models = mig_partition, models
        # logger.info(f'Evaluating {self.partition_code[mig_partition]} models {models}')                

    '''
    look through history of current hour
    '''
    def lookup_evaluated_within_hour(self, curr_hour, target_config):
        for j in range(len(self.evaluated_config[curr_hour])):
            if self.evaluated_config[curr_hour][j] == target_config:
                prev_carbon = self.eval_result['carbon'][j]
                prev_acc = self.eval_result['acc'][j]
                prev_lat = self.eval_result['lat'][j]
                prev_deltaAcc = self.eval_result['deltaAccPct'][j]
                prev_deltaCarbon = self.eval_result['deltaCarbonPct'][j]
                prev_carbonSave = self.eval_result['carbonSavePct'][j]
                return prev_acc, prev_lat, prev_carbon, prev_deltaAcc, prev_deltaCarbon, prev_carbonSave
        raise RuntimeError(f'Cannot find evaluated configuration at hour {curr_hour}')    

    '''
    Look through history of all the hours before
    '''
    def lookup_evaluated_config(self, curr_hour):
        for i in range(len(self.evaluated_config)): # i represents the hour
            for j in range(len(self.evaluated_config[i])):
                if self.evaluated_config[i][j] == (self.mig_partition, self.models):
                    prev_carbon = self.eval_result_seq[i]['carbon'][j]
                    prev_acc = self.eval_result_seq[i]['acc'][j]
                    prev_lat = self.eval_result_seq[i]['lat'][j]
                    prev_ci = self.ci_profile[i]
                    prev_pue = self.pue_profile[i]
                    new_carbon = prev_carbon * self.ci_profile[curr_hour] * self.pue_profile[curr_hour] / (prev_ci * prev_pue) 
                    return prev_acc, prev_lat, new_carbon
        raise RuntimeError('Cannot find previously evaluated configuration')    

    def run(self, Lambda, base_ci=250, base_pue=1.5):
        Path(f'{self.root_dir}/system/logs/{self.exp_algorithm}').mkdir(parents=True, exist_ok=True)    
        random.seed(self.seed)
        self.get_ci_and_pue(ci_path='csv/carbon_intensity.csv')
        self.setup_obj_func(Lambda=Lambda, ci_base=base_ci, pue_base=base_pue)
        optimized_ci, optimized_pue = 0, 0

        for i in range(len(self.ci_profile)):                 
            self.evaluated_config[i] = []
            self.eval_result = {'acc': [], 'lat': [], 'carbon': [], 
                                'reconfig_time': [], 'mig_reset': [], 'score': [],
                                'deltaAccPct': [], 'deltaCarbonPct': [], 'carbonSavePct': []
                                }
            no_improvement_iter = 0
            best_score = -1000
            best_config = (self.mig_partition, self.models)
            reset_mig = False
            logger.info(f'Optimizing for hour {i} ci={self.ci_profile[i]} pue={self.pue_profile[i]}')            

            if i > 0:
                diff = abs(self.ci_profile[i]*self.pue_profile[i] - optimized_ci*optimized_pue) / (self.avg_ci*self.avg_pue) * 100                
                if diff <= 5:
                    self.eval_result_seq.append(self.eval_result)
                    logger.info(f'No significant ci and PUE changes detected (<=10%), skip optimization')
                    continue
            optimized_ci, optimized_pue = self.ci_profile[i], self.pue_profile[i]
            baseline_carbon = self.base_unit_carbon*self.ci_profile[i]*self.pue_profile[i]            

            while True:
                logger.info(f'started evaluation of MIG {self.mig_partition} {self.models}')
                # check if this sample has already been used before
                if (self.mig_partition, self.models) in self.evaluated_config_set:
                    logger.info('used history to imply')
                    acc, lat, carbon = self.lookup_evaluated_config(i)
                    self.eval_result['acc'].append(acc)
                    self.eval_result['lat'].append(lat)
                    self.eval_result['carbon'].append(carbon)
                    reconfig_time = 0
                    self.eval_result['mig_reset'].append(False)
                else:
                    # reconfig MIG, and evaluate
                    tstart = time.perf_counter()
                    self.reconfig_mig(self.mig_partition, self.models, reset_mig)
                    reconfig_time = time.perf_counter() - tstart
                    self.eval_result['mig_reset'].append(reset_mig)
                    time.sleep(1)

                    tracker_ips = system_utils.parse_tracker_ip(self.service_name, self.root_dir)
                    req_json = {'ci': self.ci_profile[i], 'pue': self.pue_profile[i]}
                    for tracker_ip in tracker_ips:
                        system_utils.post_req(tracker_ip, 'carbon', request=req_json)
                    # start tracker
                    req_json = {'info': 'start', 'testcase': ''}
                    for tracker_ip in tracker_ips:
                        system_utils.post_req(tracker_ip, 'track', request=req_json)  
                    # start request generator, BLOCKING
                    send_signal(self.generator_ip.ip, self.generator_ip.port, 
                                cmd=f'start {self.service_name} {self.req_limit} {self.arrival}')
                    logger.info(f'finished evaluation of current configuration')
                self.eval_result['reconfig_time'].append(reconfig_time)
                self.evaluated_config[i].append((self.mig_partition, self.models))
                self.evaluated_config_set.add((self.mig_partition, self.models))
                # check for improvement
                score, deltaAcc, deltaCarbon = self.obj_func.effective_carbon_reduction_system(curr_acc=self.eval_result['acc'][-1],
                                                                        curr_carbon=self.eval_result['carbon'][-1])
                self.eval_result['score'].append(score*100)
                self.eval_result['deltaAccPct'].append(deltaAcc*100)
                self.eval_result['deltaCarbonPct'].append(deltaCarbon*100)
                carbon_save = 100 if baseline_carbon == 0 else (baseline_carbon - self.eval_result['carbon'][-1]) / baseline_carbon * 100
                self.eval_result['carbonSavePct'].append(carbon_save)

                if score > best_score and self.eval_result['lat'][-1] <= self.SLA:
                    best_score = score
                    best_config = (self.mig_partition, self.models)
                    no_improvement_iter = 0
                    logger.info(f'changed best score: {best_score}, {best_config}')
                else:
                    no_improvement_iter += 1
                logger.info(f'no improvement iteration = {no_improvement_iter}')
                # term condition
                explored_iter = sum(1 for k in self.eval_result['reconfig_time'] if k != 0)
                if (no_improvement_iter >= self.no_improv_limit or explored_iter >= self.iter_limit) and best_score != -1000:
                    if no_improvement_iter >= self.no_improv_limit:
                        logger.info(f'no improvement limit {self.no_improv_limit} is hit, terminating current hour exp')
                    if explored_iter >= self.iter_limit:
                        logger.info(f'max exploration limit {self.iter_limit} is hit, terminating current hour exp')
                    
                    # if curr_config is curr_best, do nothing and break, otherwise do another iteration
                    if (self.mig_partition, self.models) != best_config:
                        logger.info(f'changing back to {best_config} and terminating')
                        reset_mig = False if best_config[0] == self.mig_partition else True
                        self.mig_partition, self.models = best_config[0], best_config[1]
                        tstart = time.perf_counter()
                        self.reconfig_mig(self.mig_partition, self.models, reset_mig)
                        reconfig_time = time.perf_counter() - tstart
                        prev_results = self.lookup_evaluated_within_hour(i, best_config)
                        self.eval_result['acc'].append(prev_results[0])
                        self.eval_result['lat'].append(prev_results[1])
                        self.eval_result['carbon'].append(prev_results[2])
                        self.eval_result['deltaAccPct'].append(prev_results[3])
                        self.eval_result['deltaCarbonPct'].append(prev_results[4])
                        self.eval_result['carbonSavePct'].append(prev_results[5])
                        self.eval_result['reconfig_time'].append(reconfig_time)
                        self.eval_result['mig_reset'].append(reset_mig)
                        score, _, _ = self.obj_func.effective_carbon_reduction_system(curr_acc=prev_results[0],
                                                                                curr_carbon=prev_results[2])
                        self.eval_result['score'].append(score*100)
                        self.evaluated_config[i].append(best_config)                        
                        time.sleep(20)
                    else:
                        logger.info(f'current config {best_config} is the best one, terminating')
                    self.eval_result_seq.append(self.eval_result)
                    break

                mig_partition, models = self.gen_random_mig(curr_hour=i)      
                reset_mig = False if mig_partition == self.mig_partition else True
                self.mig_partition, self.models = mig_partition, models
                time.sleep(3)

            with open(f'{self.root_dir}/system/logs/{self.exp_algorithm}/{self.service_name}_eval.json', 'w') as f:
                json.dump(self.eval_result_seq, f, indent=4)
            with open(f'{self.root_dir}/system/logs/{self.exp_algorithm}/{self.service_name}_config.json', 'w') as f:
                json.dump(self.evaluated_config, f, indent=4)
        self.term_thread()
        self.listener_thread.join()

if __name__ == '__main__':
    # ins = RandomExperiment(service_name='yolo')
    exp = BaseExperiment('albert', num_nodes=1, gpus_per_node=2)
    ip1, ip2, ip3 = exp.initialize(mig_partition='0', models='3')
    exp.run()
    # exp.test_run()