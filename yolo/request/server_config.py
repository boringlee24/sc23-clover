import requests
import argparse
import json
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
import logging.config
logging.config.fileConfig(fname=f'{ROOT}/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# os.chdir(str(ROOT)) 

def send_req(args, URL):
    if args.mode == 'term':        
        req_json = {'term': ''}
    elif args.mode == 'start': # start service on MIG slices
        req_json = {'start': [args.gpuid, args.mig_config, args.weights]}
    elif args.mode == 'config': # config gpu
        req_json = {'config': [args.gpuid, args.mig_config]}
    elif args.mode == 'mix_start': # mix models on MIG device
        # the weights will be a dict, mapping each model to a mig slice
        # --weights xx6l means v5x, v5x, v5x6, v5l
        weight_list = []
        for char in args.weights:
            if char == '6':
                weight_list.append('yolov5x6')
            elif char == 'x':
                weight_list.append('yolov5x')
            elif char == 'l':
                weight_list.append('yolov5l')
            else:
                raise RuntimeError('invalid char in weight')
        req_json = {'mix_start': [args.gpuid, args.mig_config], 
                    'weights': weight_list}
    else:
        raise RuntimeError('Error: No mode specified.')

    test_response = requests.post(f'{URL}/config', json=req_json)
    if not test_response.ok:
        raise RuntimeError('GPU node master service not available')
    else:
        logger.info('configured GPU node successfully')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mig_config', type=str, default='0', help='configuration of MIG according to NVIDIA table, 0-17')    
    parser.add_argument('--gpuid', type=int, default=0, help='GPU ID (0 or 1)')  
    parser.add_argument('--weights', type=str, default='yolov5x6', help='model name: yolov5s, yolov5x, yolov5x6')
    parser.add_argument('--mode', type=str, help='operation mode: term/start/config', default='term')            
    args = parser.parse_args()

    with open(f'{str(ROOT)}/master.json') as f:
        master = json.load(f)
    URL = master['master']
    send_req(args, URL)

        


    

