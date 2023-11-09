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
import networkx as nx
from clover import CloverExperiment
import itertools
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
GITROOT=FILE.parents[1]
import logging.config
logging.config.fileConfig(fname=f'{ROOT}/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class OracleSimulator(CloverExperiment):
    def __init__(self, service_name, num_nodes=1, gpus_per_node=1, req_limit=1000):
        super().__init__(service_name, num_nodes, gpus_per_node, req_limit)

    def run(self, base_ci=1, base_pue=1):
        Path(f'{self.root_dir}/system/simulator/{self.exp_algorithm}').mkdir(parents=True, exist_ok=True)    
        save_file = Path(f'{self.root_dir}/system/simulator/{self.exp_algorithm}/{self.service_name}.json')
        if save_file.is_file():
            with open(f'{self.root_dir}/system/simulator/{self.exp_algorithm}/{self.service_name}.json') as f:
                data = json.load(f)
        else:
            data = {}
        reset_mig = True # always reset MIG in simulator profiling
        self.eval_result = {'acc': [], 'lat': [], 'carbon': []}
        idx = 0

        for graph in self.graph_to_partition:
            idx += 1
            if graph not in data:
                data[graph] = {}
                self.mig_partition, self.models = self.graph_to_partition[graph]
                logger.info(f'started evaluation of MIG {self.mig_partition} {self.models}')

                tstart = time.perf_counter()
                self.reconfig_mig(self.mig_partition, self.models, reset_mig)
                reconfig_time = time.perf_counter() - tstart
                data[graph]['reconfig_time'] = reconfig_time
                time.sleep(1)

                tracker_ips = system_utils.parse_tracker_ip(self.service_name, self.root_dir)
                req_json = {'ci': base_ci, 'pue': base_pue}
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
                data[graph]['acc'] = self.eval_result['acc'][-1]
                data[graph]['lat'] = self.eval_result['lat'][-1]
                data[graph]['carbon'] = self.eval_result['carbon'][-1]

                if idx % 10 == 0:
                    logger.info(f'process: {idx} out of {len(self.graph_to_partition)}')
                    with open(f'{self.root_dir}/system/simulator/{self.exp_algorithm}/{self.service_name}.json', 'w') as f:
                        json.dump(data, f, indent=4)
                time.sleep(10)
                
        with open(f'{self.root_dir}/system/simulator/{self.exp_algorithm}/{self.service_name}.json', 'w') as f:
            json.dump(data, f, indent=4)
        logger.info('finished')
        self.term_thread()
        self.listener_thread.join()

if __name__ == '__main__':
    exp = OracleSimulator('efficientnet', num_nodes=1, gpus_per_node=2, req_limit=1000)
    exp.exp_algorithm = 'exhaustive'
    exp.initialize(mig_partition='0', models='7')
    exp.run()