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

Experiments are run with Python 1.10, CUDA 11.7, NVIDIA A100 GPUs

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

The instructions below are based on our experiment node with 2 A100 GPUs.

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
Depending on the application, choose one of the actions from below

###### Albert
```
cd albert 
python setup_service.py
```

4. Start Clover controller

In a CPU node, start the controller service

```
python
```

5. Start the request generator

In another CPU node (different from the controller node), start the inference requests
```
python
```

_____

If ``run_mix.py`` is available, don't need to configure the node below, just run this script. Otherwise, follow below:

3. From a job submitter node, configure the GPU node
```
cd request
python server_config.py --mig_config XX --mode config
```
Then start the inference applications.
```
python server_config.py --mig_config XX --weights XX --mode start/mix_start
```

Wait for the inference and carbon tracker applications to start.

4. From the job submitter node, start the inference request
```
python request.py --inf_num XX --testcase XX
```
or if used mixed models in one GPU:
```
python request_mix.py --inf_num XX --testcase XX
```
