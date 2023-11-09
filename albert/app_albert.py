import io
from PIL import Image
import PIL
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
from torchvision import transforms
from transformers import AlbertTokenizer, AlbertForQuestionAnswering

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
util_path = f'{str(ROOT)}/yolo'
if util_path not in sys.path:
    sys.path.append(util_path)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from utils.dataloaders import IMG_FORMATS, LoadImages
from utils.general import print_args
# from utils.torch_utils import select_device, time_sync
# from models.common import DetectMultiBackend

os.environ['TORCH_HOME'] = '/work/li.baol/.cache/torch/'
# huggingface pretrained models
model_mapping = {
    'base': 'twmkn9/albert-base-v2-squad2',
    'large': 'elgeish/cs224n-squad2.0-albert-large-v2',
    'xlarge': 'ktrapeznikov/albert-xlarge-v2-squad-v2',
    'xxlarge': 'mfeb/albert-xxlarge-v2-squad2'
}

app = Flask(__name__)

# load the model and warm it up
def run(args):
    # # Load model
    hf_model = model_mapping[args.weights]
    tokenizer = AlbertTokenizer.from_pretrained(hf_model)    
    model = AlbertForQuestionAnswering.from_pretrained(hf_model).to('cuda').eval()

    question = "When were the Normans in Normandy?" 
    context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."

    inputs = tokenizer(question, context, return_tensors="pt").to('cuda')

    # model.eval()
    with torch.no_grad():
        result = model(**inputs)
    return model, tokenizer

@app.route('/')
@app.route('/index.html')
def index():
    return f'<p>Hosting service albert_{args.weights}_v2 at port {args.port}</p>'

DETECTION_URL = '/detect'

# accepts both types of request: file or json body
@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return

    if request.is_json:
        # pdb.set_trace()
        req_json = request.get_json()
        inputs = tokenizer(req_json['question'], req_json['context'], return_tensors='pt', 
                            truncation=True, max_length=512).to('cuda') # the max allowed length of this model is 512
    else:
        raise RuntimeError('Only json is accepted as input')
    
    with torch.no_grad():
        pred = model(**inputs)
    
    answer_start_index = pred.start_logits.argmax()
    answer_end_index = pred.end_logits.argmax()    
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    response = tokenizer.decode(predict_answer_tokens)

    output = {
        'model': f'albert_{args.weights}_v2',
        'answer': response,
        'batch': 1
    }    
    return make_response(jsonify(output), 201)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='base', help='model name: albert base, large, ...')
    parser.add_argument('--port', type=int, default=5000, help='different port for different MIG device')
    args = parser.parse_args()
    # args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    print_args(vars(args))
    return args    

if __name__ == '__main__':

    args = parse()

    model, tokenizer = run(args)

    app.run(debug=False,host='0.0.0.0',port=args.port, threaded=True)
