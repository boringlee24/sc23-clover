import io
from PIL import Image
import torch
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from utils.dataloaders import IMG_FORMATS, LoadImages
from utils.general import print_args
# from utils.torch_utils import select_device, time_sync
# from models.common import DetectMultiBackend

app = Flask(__name__)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# load the model and warm it up
def run(args):
    # # Load model
    # # model = torch.hub.load(f'{ROOT}/{args.model}.pt', args.model, pretrained='True')#, source='local')#, pretrained='True')
    # device = select_device(device)
    # model = DetectMultiBackend(f'{ROOT}/{weights}.pt', device=device, fp16=half)
    # model.warmup(imgsz=(1, 3, *imgsz))  # warmup    
    model = torch.hub.load(repo_or_dir=f'{ROOT}', model='custom', source='local', path=f'{ROOT}/{args.weights}.pt', force_reload=True)
    if torch.cuda.is_available():
        model.half()
    # warm up the model
    if args.weights == 'yolov5x6':
        size = 1280
    else:
        size = 640

    result = model('/work/li.baol/GIT/datasets/coco/images/val2017/000000000139.jpg', size=size)
    return model

def tensor2str(tsr):
    x = tsr.cpu().numpy().astype(np.float64) #detach().
    # x_rnd = np.around(x, decimals=4)    
    # x_new = copy.deepcopy(x_rnd)
    x = x.round(decimals=2)
    x = json.dumps(x, cls=NumpyArrayEncoder)
    return x

@app.route('/')
@app.route('/index.html')
def index():
    return f'<p>Hosting service {args.weights} at port {args.port}</p>'

DETECTION_URL = '/detect'

# accepts both types of request: file or json body
@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return

    if request.is_json:
        # pdb.set_trace()
        im_batch = []
        req_json = request.get_json()
        for item in req_json['image_list']:
            im_bytes = bytes.fromhex(item)
            im = Image.open(io.BytesIO(im_bytes))
            im_batch.append(im)              

    elif request.files.get('img'):
        # Method 1
        # with request.files['image'] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_files = request.files.getlist('img')
        im_batch = []
        for im_file in im_files:
            im_bytes = im_file.read()
            im = Image.open(io.BytesIO(im_bytes))
            im_batch.append(im)        

    if args.weights == 'yolov5x6':
        size = 1280
    else:
        size = 640
    
    pred = model(im_batch, size=size)  # reduce size=320 for faster inference
    
    response = [tensor2str(x) for x in pred.xyxy]
    # # To decode:
    # np_array = np.asarray(json.loads(encodedNumpyData))

    output = {
        'model': args.weights,
        'bbox': response,
        'batch': len(response)
    }    
    return make_response(jsonify(output), 201)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5x6', help='model name: yolov5s, yolov5x, yolov5x6')
    parser.add_argument('--port', type=int, default=5000, help='different port for different MIG device')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    args = parser.parse_args()
    # args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    print_args(vars(args))
    return args    

if __name__ == '__main__':

    args = parse()

    model = run(args)
    
    app.run(debug=False,host='0.0.0.0',port=args.port, threaded=True)
