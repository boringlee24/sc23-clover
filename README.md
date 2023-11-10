# Clover: Toward Sustainable AI with Carbon-Aware Machine Learning Inference Service

Paper at 2023 ACM/IEEE The International Conference for High Performance Computing, Networking, Storage, and Analysis (SC'23)

### Citation

```
@inproceedings{li2023clover,
  title={Clover: Toward Sustainable AI with Carbon-Aware Machine Learning Inference Service},
  author={Li, Baolin and Samsi, Siddharth and Gadepally, Vijay and Tiwari, Devesh},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--15},
  year={2023}
}
```

### Dependencies

Experiments are run with Python 3.10, CUDA 11.7, NVIDIA A100 GPUs

1. Build and install Clover's carbon monitor 

clone https://github.com/boringlee24/power_monitor.git, then navigate into the cloned directory, run

```
python setup.py install
```

Test if the carbon monitor is successfully installed by running

```
python -c "import carbontracker"
```

2. Install the rest of the python packages

```
pip install -r requirements.txt
```

### Setup

The instructions below are based on our experiment node with 2 A100 GPUs. Make sure you have sudo access.

The service logs will be dumped to ``/scratch/li.baol/carbon_logs/`` by default. Create this directory or change all instances of this directory to your own log directory.

#### 1. Set up controller node

On a CPU node, start the controller service. Use one of the following for ``--service``: yolo, efficientnet, albert.

```
cd system
python run.py --service yolo
```
Note: the controller node must be started before the GPU node and request node.

A log file will be generated in ``run.log`` that records important events during the experiment.

#### 2. Set up GPU node

1. Enable MIG in ``mig`` directory
```
cd mig
python mig_helper.py --init --gpu 0
python mig_helper.py --init --gpu 1
```

2. Generate the MIG device lookup table (maps from MIG slice to UUID)

```
python export_cuda_device_auto.py --num_gpus 2
```

3. Run the GPU node service
```
cd ../
```
Depending on the application, choose one of the actions from below. Make sure it matches the controller node's ``--service`` argument.

###### Albert
```
cd albert
python setup_service.py --system
```
###### EfficientNet
We use ImageNet validation dataset. Download the dataset [here](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).

Assuming the parent directory of this repository is ``<GIT_DIR>``, the application will look in the following directory ``<GIT_DIR>/efficient-net/examples/imagenet/data/val/`` to load the Imagenet validation dataset. Download the datasets and put them in the corresponding directory. 

```
cd efficientnet
python setup_service.py --system
```

###### YOLO
We use the COCO validation dataset. Download the dataset [here](https://cocodataset.org/#download).

The local COCO validation dataset path used by this project is ``<GIT_DIR>/datasets/coco/images/val2017/``.

```
cd yolo
python setup_service.py --system
```

#### 3. Start inference requests

On a CPU node that is different from the controller node, start the request service. 
```
cd system
python generator.py
```

### 4. Wait for the experiment to finish

Go back to the controller node. Monitor the ``run.log`` file for events during the experiment. If it gets stuck waiting, e.g., ``Waiting for service URLs to be up``, check the ``/scratch/li.baol/carbon_logs/`` directory for error messages.

Once completed, the controller node will print out the saved carbon emission and related information.

### Contact

Baolin Li: https://baolin-li.netlify.app/

### Carbon Footprint Characterization

In addition to the carbon-aware inference scheduler, we also conducted a comprehensive study on the carbon footprint of HPC systems. See here for [[Code](https://github.com/boringlee24/sc23-sustainability)], [[Paper](https://dl.acm.org/doi/10.1145/3581784.3607035)].