import socket
from pathlib import Path
import os, sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] 
MIG_DIR = f'{str(ROOT)}/mig'
SYS_DIR = f'{str(ROOT)}/system'
sys.path.append(MIG_DIR)
sys.path.append(SYS_DIR)
from system_utils import send_signal
import mig_helper
import argparse
import json
import subprocess
from flask import Flask, request, make_response, jsonify, make_response
import pdb
import psutil
import time

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return f'<p>Hosting GPU node master service at port {args.port}</p>'

URL='/config'
output = {'state': 'success'}
@app.route(URL, methods=['POST'])
def configure():
    # {'term': '', 'config': [gpuid, mig_config], 
    # 'start': [gpuid, mig_config, weights]}
    if request.method != 'POST' or not request.is_json:
        return
    req_json = request.get_json() # {'info': 'start'/'end'}

    # terminate the worker services
    if 'term' in req_json:
        procs = psutil.Process().children()
        for p in procs:
            p.terminate()
        gone, alive = psutil.wait_procs(procs, timeout=30)
        if len(alive) > 0:
            for p in alive:
                print(p.pid)
            raise RuntimeError('The above process cannot be terminated')
        return make_response(jsonify(output), 201)
    
    elif 'config' in req_json:
        gpuid, mig_config = req_json['config']
        config_gpu(gpuid, mig_config)
        return make_response(jsonify(output), 201)
    elif 'multi_config' in req_json:
        num_gpu, mig_config = req_json['multi_config']
        for gpuid in range(num_gpu):
            config_gpu(gpuid, mig_config)
        return make_response(jsonify(output), 201)
    elif 'mix_start' in req_json:
        gpuid, mig_config = req_json['mix_start']
        weight_list = req_json['weights']
        start_mixed_service(gpuid, mig_config, weight_list)
        return make_response(jsonify(output), 201)
    elif 'multi_start' in req_json:
        num_gpu, mig_config = req_json['multi_start']
        weight_list = req_json['weights']
        print(req_json)
        start_multigpu_service(num_gpu, mig_config, weight_list)
        return make_response(jsonify(output), 201)

# configure the GPU
def config_gpu(gpuid, mig_config):
    # reset the GPU
    mig_helper.reset_mig(gpuid)

    # partition to desired config
    with open(f'{MIG_DIR}/partition_code.json') as f:
        partition = json.load(f)
    mig_helper.do_partition(gpuid, partition[mig_config])

def start_multigpu_service(num_gpu, mig_config, weight_list):
    with open(f'{MIG_DIR}/partition_code.json') as f:
        partition = json.load(f)
    if len(weight_list) != len(partition[mig_config]):
        raise RuntimeError('Number of models =/= number of slices')
    with open(f'{MIG_DIR}/mig_device_autogen.json') as f:
        device_json = json.load(f)

    tracker_ports = {}
    service_ports = {} # {pid: [model_slice, url]}
    idx = 0
    for k in range(num_gpu):
        mig_list = device_json[hostname][f'gpu{k}'][mig_config]
        for i, (device, weight) in enumerate(zip(mig_list, weight_list)):
            cmd = f'CUDA_VISIBLE_DEVICES={device} python app_albert.py --weights={weight} --port 500{idx}'        
            out_file = f'/scratch/li.baol/carbon_logs/albert{idx}.out'
            err_file = f'/scratch/li.baol/carbon_logs/albert{idx}.err'
            with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                proc = subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
            service_ports[proc.pid] = [f'{weight}_{idx}', f'http://{ip_addr}:500{idx}']
            idx += 1            
            time.sleep(0.1)

    # start the app_track.py
    cmd = f'python app_track.py --port 5100'
    out_file = f'/scratch/li.baol/carbon_logs/tracker.out'
    err_file = f'/scratch/li.baol/carbon_logs/tracker.err'
    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
        proc = subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
    tracker_ports[proc.pid] = f'http://{ip_addr}:5100'

    # log to json
    with open(f'service.json', 'w') as f:
        json.dump(service_ports, f, indent=4)
    with open(f'tracker.json', 'w') as f:
        json.dump(tracker_ports, f, indent=4)      

def start_mixed_service(gpuid, mig_config, weight_list):
    with open(f'{MIG_DIR}/partition_code.json') as f:
        partition = json.load(f)
    if len(weight_list) != len(partition[mig_config]):
        raise RuntimeError('Number of models =/= number of slices')
    with open(f'{MIG_DIR}/mig_device_autogen.json') as f:
        device_json = json.load(f)

    mig_list = device_json[hostname][f'gpu{gpuid}'][mig_config]       
    service_ports = {} # {pid: [model_slice, url]}
    tracker_ports = {}
    for i, (device, weight) in enumerate(zip(mig_list, weight_list)):
        cmd = f'CUDA_VISIBLE_DEVICES={device} python app_albert.py --weights={weight} --port 500{i}'        
        out_file = f'/scratch/li.baol/carbon_logs/albert{i}.out'
        err_file = f'/scratch/li.baol/carbon_logs/albert{i}.err'
        with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
            proc = subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
        service_ports[proc.pid] = [f'{weight}_{i}', f'http://{ip_addr}:500{i}']

    # start the app_track.py
    cmd = f'python app_track.py --port 5100'
    out_file = f'/scratch/li.baol/carbon_logs/tracker.out'
    err_file = f'/scratch/li.baol/carbon_logs/tracker.err'
    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
        proc = subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
    tracker_ports[proc.pid] = f'http://{ip_addr}:5100'

    # log to json
    with open(f'service.json', 'w') as f:
        json.dump(service_ports, f, indent=4)
    with open(f'tracker.json', 'w') as f:
        json.dump(tracker_ports, f, indent=4)      

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5200, help='port for gpu node master server')
    parser.add_argument('--system', action='store_true', help='run in system mode', default=False)
    args = parser.parse_args()
    # args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    print(args)
    return args    

if __name__ == '__main__':
    print('creating services on MIG devices')

    args = parse()
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    master_ports = {'master': f'http://{ip_addr}:{args.port}'}
    with open(f'master.json', 'w') as f:
        json.dump(master_ports, f, indent=4)     
    if args.system:
        with open(f'{str(ROOT)}/system/controller.json') as f:
            controller_ip = json.load(f)
        send_signal(controller_ip['ip'], controller_ip['port'], cmd=f'gpu_node ip {ip_addr}')
    app.run(debug=False,host='0.0.0.0',port=args.port)

    



