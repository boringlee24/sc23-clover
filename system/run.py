from experiment import BaseExperiment
from clover import CloverExperiment
import time
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--service', type=str, help="application: albert/yolo/efficientnet", default='yolo')
parser.add_argument('--num_nodes', type=int, default=1)
parser.add_argument('--gpus_per_node', type=int, default=2)
parser.add_argument('--init_mig', type=str, help='initial mig partition', default='0')    
parser.add_argument('--Lambda', type=float, help='Lambda value of obj func', default=0.1)
parser.add_argument('--num_req', type=int, help="# of inference requests to evaluate Clover config", default=1000)
parser.add_argument('--skip', action='store_true', help="Set to True to skip experiment, directly process data", default=False)
args = parser.parse_args()

init_models = {'yolo': '6', 'albert': '3', 'efficientnet': '7'} # yolov5x6, albert-xxlarge, efficientnet-b7

if not args.skip:
    print('Starting Baseline experiments')
    exp = BaseExperiment(args.service, num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)
    ip1, ip2, ip3 = exp.initialize(mig_partition=args.init_mig, models=init_models[args.service])
    exp.run()
    print('Finished Baseline')

    time.sleep(30)

    print('Starting Clover experiments')
    exp = CloverExperiment(args.service, 
                        num_nodes=args.num_nodes, 
                        gpus_per_node=args.gpus_per_node, 
                        req_limit=args.num_req)

    exp.initialize(mig_partition=args.init_mig, 
                models=init_models[args.service], 
                wait=False, 
                gpu_node_ip=ip1, 
                generator_ip=ip2, 
                generator_port=ip3)

    exp.run(Lambda=args.Lambda)
    print('finished clover')

# Analyze the carbon savings compared to baseline

scheme = 'clover'
service = args.service
Lambda = 0.1
eval_time = 30 # 30seconds

metrics = ['acc', 'carbon', 'lat']
data = {}

for mi, metric in enumerate(metrics):
    data[metric] = {}
    data_metric = data[metric]
    data_metric['base'], data_metric[scheme] = [], []
    with open(f'logs/base/{service}.json') as f:
        base = json.load(f)        
    data_metric['base'].append(np.mean(base[metric]))

    with open(f'logs/{scheme}/{service}_eval.json') as f:
        read = json.load(f)
    read_list = []
    valid_hr = 0
    for hour, hourly_data in enumerate(read):
        if len(hourly_data['acc']) == 0: # use previous hour
            read_list.append(read[valid_hr][metric][-1])
        else:
            weight_seconds = list(map(lambda x: eval_time if x!=0 else 0, hourly_data['reconfig_time'][:-1]))
            last_weight = 3600 - sum(weight_seconds)
            weight_seconds.append(last_weight)
            if metric == 'lat':
                read_list.append(hourly_data[metric][-1])
            read_list.append(np.average(hourly_data[metric], weights=weight_seconds))
            valid_hr = hour
    data_metric[scheme].append(np.mean(read_list))
    if metric == 'carbon':    
        data_metric['norm'] = (1 - np.array(data_metric[scheme]) / np.array(data_metric['base']))*100
    elif metric == 'acc':
        data_metric['norm'] = (np.array(data_metric['base']) - np.array(data_metric[scheme]))
    else:
        data_metric['norm'] = np.array(data_metric[scheme]) / np.array(data_metric['base'])

print(f"Application: {service}")
print(f"Clover carbon saving: {data['carbon']['norm'][0]}%")
print(f"Clover accuracy loss: {data['acc']['norm'][0]}%")
print(f"Clover latency drop: {(1-data['lat']['norm'][0])*100}%")
