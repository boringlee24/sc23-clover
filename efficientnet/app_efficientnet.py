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
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

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
app = Flask(__name__)

# load the model and warm it up
def run(args):
    # # Load model
    model = EfficientNet.from_pretrained(f'efficientnet-{args.weights}').cuda().eval()
    # model = model.cuda() # move model into GPU (CUDA_VISIBLE_DEVICE is set by service starter)
    image_size = EfficientNet.get_image_size(f'efficientnet-{args.weights}')

    img_transform = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC), 
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    # warm up the model
    img1 = Image.open('/work/li.baol/GIT/efficient-net/examples/imagenet/data/val/n02101388/ILSVRC2012_val_00015972.JPEG')
    img2 = Image.open('/work/li.baol/GIT/efficient-net/examples/imagenet/data/val/n02101388/ILSVRC2012_val_00009088.JPEG')
    img1 = img_transform(img1)#.unsqueeze(0)
    img2 = img_transform(img2)
    img = torch.stack([img1, img2], dim=0).cuda() #img1.cuda()

    # model.eval()
    with torch.no_grad():
        result = model(img)
    return model, img_transform

@app.route('/')
@app.route('/index.html')
def index():
    return f'<p>Hosting service efficientnet-{args.weights} at port {args.port}</p>'

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
        im_files = request.files.getlist('img')
        im_batch = []
        for im_file in im_files:
            im_bytes = im_file.read()
            im = Image.open(io.BytesIO(im_bytes))
            if im.mode != 'RGB':
                im = im.convert('RGB') # if picture is black/white, convert to RGB
            im_batch.append(im)     

    imgs = [img_transform(k) for k in im_batch]
    input = torch.stack(imgs, dim=0).cuda()
    
    with torch.no_grad():
        pred = model(input)
    
    response = torch.argmax(pred, dim=1).cpu().numpy().tolist()
    # # To decode:
    # np_array = np.asarray(json.loads(encodedNumpyData))

    output = {
        'model': f'efficientnet-{args.weights}',
        'prediction': response,
        'batch': len(response)
    }    
    return make_response(jsonify(output), 201)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='b0', help='model name: efficientnet b0, b1, ...')
    parser.add_argument('--port', type=int, default=5000, help='different port for different MIG device')
    args = parser.parse_args()
    # args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    print_args(vars(args))
    return args    

if __name__ == '__main__':

    args = parse()

    model, img_transform = run(args)

    app.run(debug=False,host='0.0.0.0',port=args.port, threaded=True)
