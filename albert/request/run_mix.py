import subprocess
from pathlib import Path
import os, sys
FILE = Path(__file__).resolve()
MIG = FILE.parents[2] 
ALBERT = FILE.parents[1]
MIG_DIR = f'{str(MIG)}/mig'
import json
import itertools
from collections import Counter
import logging
import logging.config
from server_config import send_req
import argparse
import time

logging.config.fileConfig(fname=f'{ALBERT}/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--mig_config', type=str, default='0', help='configuration of MIG according to NVIDIA table, 0-17')    
parser.add_argument('--gpuid', type=int, default=0, help='GPU ID (0 or 1)')  
parser.add_argument('--mode', type=str, help='operation mode: term/start/config', default='term')            
parser.add_argument('--weights', type=str, help='weights', default='term')            
args = parser.parse_args()

with open(f'{ALBERT}/master.json') as f:
    master = json.load(f)
URL = master['master']

def get_model_code(input):
    s = ''
    for model in input:
        s += str(models.index(model))
    return s

models = ['base', 'large', 'xlarge', 'xxlarge']

partitions = [0, 1, 3, 4, 7, 10, 17]
#partitions = [10]

with open(f'{MIG_DIR}/partition_code.json') as f:
    code = json.load(f)

args.mode = 'term'
send_req(args, URL)

for p in partitions:
    slices = code[str(p)]
    counter_list = [] # this avoids repeating the same configuration
    # generate all combination of models
    for config in itertools.product(models, repeat=len(slices)):
        map_list = []
        for s, c in zip(slices, config):
            map_list.append(f'{c}_slice{s}')
        counter = Counter(map_list)
        if counter in counter_list:
            continue
        counter_list.append(counter)
        model_code = get_model_code(config)
        # run this 
        logger.info(f'running models {model_code} on MIG {slices}')
        # setup
        args.mig_config, args.mode = str(p), 'config'
        send_req(args, URL)
        time.sleep(10)
        args.weights, args.mode = model_code, 'mix_start'
        send_req(args, URL)
        time.sleep(30)
        logger.info('Setup done, starting inference')
        cmd = f'python request_mix.py --inf_num 2000 --testcase p{p}_{model_code}'
        subprocess.check_call([cmd], shell=True)
        # p.wait()
        logger.info('Finished, starting next testcase')
        time.sleep(3)
        args.mode = 'term'
        send_req(args, URL)
        time.sleep(3)
