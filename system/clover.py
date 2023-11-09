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
import networkx as nx
from copy import deepcopy
from sklearn.metrics.pairwise import manhattan_distances
import itertools
from experiment import RandomExperiment
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
GITROOT=FILE.parents[1]
import logging.config
logging.config.fileConfig(fname=f'{ROOT}/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class CloverExperiment(RandomExperiment):

    def __init__(self, service_name, num_nodes=1, gpus_per_node=1, req_limit=1000):
        super().__init__(service_name, num_nodes, gpus_per_node)
        self.exp_algorithm = 'clover'
        self.neighbor_thres = 4 # manhattan distance # TODO: try thres=3 also
        self.no_neighbor_limit = 400 # if cannot generate a neighbor after 400 tries, terminate
        self.sim_anneal = system_utils.SimulatedAnneling()
        self.req_limit = req_limit
        self.evaluated_config_dump = {} 
        self.mig_slices = ['7g', '4g', '3g', '2g', '1g']

        self.base_graph = nx.DiGraph()
        for v in self.possible_models[service_name] + self.mig_slices:
            self.base_graph.add_node(v)
        for vL in self.possible_models[service_name]:
            for vR in self.mig_slices:
                self.base_graph.add_edge(vL, vR, weight=0)                
        # generate reverse lookup from mig_graph to mig_partition and models
        self.graph_to_partition = {}
        for mig_partition in self.possible_partitions:
            slots = len(self.partition_code[mig_partition])            
            Cartesian_prod = list(itertools.product(self.possible_models[self.service_name], repeat=slots))
            for prod in Cartesian_prod:
                models = ''
                for i in prod:
                    models += i
                matched_graph = self.to_mig_graph(mig_partition=mig_partition, models=models)
                if matched_graph not in self.graph_to_partition:
                    self.graph_to_partition[matched_graph] = (mig_partition, models)

    def gen_random_mig(self, curr_hour):
        mig_partition = random.choice(self.possible_partitions)
        models = ''        
        for i in range(len(self.partition_code[mig_partition])):
            models += random.choice(self.possible_models[self.service_name])          
        # check if already seen in current iter
        graph_represent = self.to_mig_graph(mig_partition=mig_partition, models=models)
        if graph_represent in self.evaluated_config[curr_hour]:
            graph_represent = self.gen_random_mig(curr_hour)
        return graph_represent

    '''
    check if graph_represent is a neighbor of center. Both are strings of dicts
    '''
    def is_neighbor(self, center, graph_represent):
        center_vector = list(eval(center).values())
        graph_vector = list(eval(graph_represent).values())
        distance = manhattan_distances([center_vector], [graph_vector])
        if distance <= self.neighbor_thres:
            return True
        else:
            return False

    def to_mig_graph(self, mig_partition, models):        
        partition_code = self.partition_code[mig_partition]
        new_graph = deepcopy(self.base_graph)
        for i in range(len(partition_code)):
            vR = f'{partition_code[i]}g'
            vL = models[i]
            curr_weight = new_graph[vL][vR]['weight']
            new_graph.edges[vL,vR].update(weight=curr_weight+1)
        return str(nx.get_edge_attributes(new_graph,'weight'))      

    @property
    def mig_graph(self):
        return self.to_mig_graph(self.mig_partition, self.models)

    def lookup_evaluated_config(self, curr_hour):
        for i in range(len(self.evaluated_config)): # i represents the hour
            for j in range(len(self.evaluated_config[i])):                
                if self.evaluated_config[i][j] == self.mig_graph:
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
            self.evaluated_config_dump[i] = []
            self.eval_result = {'acc': [], 'lat': [], 'carbon': [], 
                                'reconfig_time': [], 'mig_reset': [], 'score': [],
                                'deltaAccPct': [], 'deltaCarbonPct': [], 'carbonSavePct': []
                                }
            no_improvement_iter = 0
            best_score = -1000
            best_config = self.mig_graph
            reset_mig = False
            all_neighbors_evaluated = False 
            logger.info(f'Optimizing for hour {i} ci={self.ci_profile[i]} pue={self.pue_profile[i]}')            

            if i > 0:
                diff = abs(self.ci_profile[i]*self.pue_profile[i] - optimized_ci*optimized_pue) / (self.avg_ci*self.avg_pue) * 100                
                if diff <= 5:
                    self.eval_result_seq.append(self.eval_result)
                    logger.info(f'No significant ci and PUE changes detected (<=10%), skip optimization')
                    continue
            optimized_ci, optimized_pue = self.ci_profile[i], self.pue_profile[i]
            baseline_carbon = self.base_unit_carbon*self.ci_profile[i]*self.pue_profile[i]            

            center = self.mig_graph            
            self.sim_anneal.initialize(center_score=-1000)
            logger.info('reset current center score and init temperature')
            while True:
                logger.info(f'started evaluation of MIG {self.mig_partition} {self.models}')
                # check if this sample has already been used before
                if self.mig_graph in self.evaluated_config_set:
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
                self.evaluated_config[i].append(self.mig_graph)
                self.evaluated_config_dump[i].append((self.mig_partition, self.models))
                self.evaluated_config_set.add(self.mig_graph)
                # check for improvement
                score, deltaAcc, deltaCarbon = self.obj_func.effective_carbon_reduction_system(curr_acc=self.eval_result['acc'][-1],
                                                                        curr_carbon=self.eval_result['carbon'][-1])
                self.eval_result['score'].append(score*100)
                self.eval_result['deltaAccPct'].append(deltaAcc*100)
                self.eval_result['deltaCarbonPct'].append(deltaCarbon*100)
                carbon_save = 100 if baseline_carbon == 0 else (baseline_carbon - self.eval_result['carbon'][-1]) / baseline_carbon * 100
                self.eval_result['carbonSavePct'].append(carbon_save)
                if self.eval_result['lat'][-1] <= self.SLA:
                    new_sa_score = score
                else:
                    new_sa_score = score * self.SLA / self.eval_result['lat'][-1]
                # determine if we want to move the center
                move_center = self.sim_anneal.move_center(new_score=new_sa_score, logger=logger)
                if move_center:
                    logger.info(f'moved center to ({self.mig_partition}, {self.models})')
                    center = self.mig_graph

                if score > best_score and self.eval_result['lat'][-1] <= self.SLA:
                    best_score = score
                    best_config = self.mig_graph
                    no_improvement_iter = 0
                    logger.info(f'changed best score: {best_score}, ({self.mig_partition}, {self.models})')
                else:
                    no_improvement_iter += 1
                logger.info(f'no improvement iteration = {no_improvement_iter}')
                
                # generate a new neighbor, if cannot set all_neighbors_evaluate = True
                for k in range(self.no_neighbor_limit):
                    new_graph = self.gen_random_mig(i)
                    if self.is_neighbor(center, new_graph):
                        break
                    if k == self.no_neighbor_limit-1:
                        logger.info('cannot find a new neighbor that is not evaluated')
                        all_neighbors_evaluated = True                        

                # term condition
                explored_iter = sum(1 for k in self.eval_result['reconfig_time'] if k != 0)
                if (no_improvement_iter >= self.no_improv_limit or explored_iter >= self.iter_limit or all_neighbors_evaluated) and best_score != -1000:
                    if all_neighbors_evaluated:
                        all_neighbors_evaluated = False      
                    if no_improvement_iter >= self.no_improv_limit:
                        logger.info(f'no improvement limit {self.no_improv_limit} is hit, terminating current hour exp')
                    if explored_iter >= self.iter_limit:
                        logger.info(f'max exploration limit {self.iter_limit} is hit, terminating current hour exp')
                    # if curr_config is curr_best, do nothing and break, otherwise do another iteration
                    if self.mig_graph != best_config:
                        logger.info(f'changing back to {self.graph_to_partition[best_config]} and terminating')
                        reset_mig = False if self.graph_to_partition[best_config][0] == self.mig_partition else True
                        self.mig_partition, self.models = self.graph_to_partition[best_config]
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
                        self.evaluated_config_dump[i].append((self.mig_partition, self.models))    
                        time.sleep(20)
                    else:
                        logger.info(f'current config {self.graph_to_partition[best_config]} is the best one, terminating')
                    self.eval_result_seq.append(self.eval_result)
                    break

                mig_partition, models = self.graph_to_partition[new_graph]
                reset_mig = False if mig_partition == self.mig_partition else True
                self.mig_partition, self.models = mig_partition, models
                time.sleep(3)

            with open(f'{self.root_dir}/system/logs/{self.exp_algorithm}/{self.service_name}_eval.json', 'w') as f:
                json.dump(self.eval_result_seq, f, indent=4)
            with open(f'{self.root_dir}/system/logs/{self.exp_algorithm}/{self.service_name}_config.json', 'w') as f:
                json.dump(self.evaluated_config_dump, f, indent=4)
        self.term_thread()
        self.listener_thread.join()

if __name__ == '__main__':
    exp = CloverExperiment('yolo', num_nodes=1, gpus_per_node=2, req_limit=500)
    exp.exp_algorithm = 'test_clover'
    exp.initialize(mig_partition='7', models='xxx')
    exp.run(Lambda=0.1)