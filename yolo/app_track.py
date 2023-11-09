from re import M
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
import os, sys
import argparse
from pathlib import Path
import pdb
import numpy as np
from flask import jsonify
from flask import make_response
import json
from json import JSONEncoder
import copy
sys.path.append('/work/li.baol/GIT/power_monitor')
from carbontracker.tracker import CarbonTracker, CarbonTrackerManual
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return f'<p>Hosting carbon tracking service at port {args.port}</p>'

DETECTION_URL = '/track'
CARBON_CONFIG_URL = '/carbon'
testcase = ''
tracker = CarbonTrackerManual(epochs=1, monitor_epochs=1, update_interval=1,
        components='all', epochs_before_pred=1, verbose=2)

@app.route(CARBON_CONFIG_URL, methods = ['POST'])
def carbon_config():
    global ci_manual, pue_manual, tracker
    if request.method != 'POST' or not request.is_json:
        return
    req_json = request.get_json() # {'ci': xx, 'pue': xx}
    ci_manual = req_json['ci']
    pue_manual = req_json['pue']
    tracker.intensity_updater.ci_manual = ci_manual
    tracker.tracker.pue_manual = pue_manual
    output = {
        'state': 'success'
    }    
    return make_response(jsonify(output), 201)

# accepts both types of request: file or json body
@app.route(DETECTION_URL, methods=['POST'])
def track():
    global testcase, tracker
    if request.method != 'POST' or not request.is_json:
        return

    req_json = request.get_json() # {'info': 'start'/'end'}
    output = {
        'state': 'success',
        'ci': tracker.intensity_updater.ci_manual,
        'pue': tracker.tracker.pue_manual,
    }    
    if req_json['info'] == 'start':
        # tracker = CarbonTracker(epochs=1, monitor_epochs=1, update_interval=1,
        # components='all', epochs_before_pred=1)

        tracker.epoch_start()
        testcase = req_json['testcase']
    elif req_json['info'] == 'end':
        tracker.epoch_end(f'{str(ROOT)}/logs/carbon/{testcase}')
    elif req_json['info'] == 'end_return':
        energy, co2 = tracker.epoch_end('')
        output['energy'] = energy
        output['co2'] = co2

    return make_response(jsonify(output), 201)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5100, help='port for carbon tracker')
    args = parser.parse_args()
    return args    

if __name__ == '__main__':

    args = parse()

    app.run(debug=False,host='0.0.0.0',port=args.port)
