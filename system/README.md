# System Design

``analysis/``: this includes hypothetical analysis based on the collected data on single GPUs

``gen_request.py``: generates requests in a thread. Operates like a queue. Producer puts request in queue, consumer sends the request to available service IP address. The round-trip time is recorded for every request individually. This must run from a requester node different from the controller and GPU nodes.

#TODO: now developing single GPU. Need to extend to GPU duplication (support multiple masters).

## Set up

### Controller node

On the controller node, run 
```
python run.py
```
This starts the run steps, and keeps a background thread (port 10002) listening to signals from GPU node and generator node
Start the controller node before other nodes

### GPU node

Go to ``mig`` folder and run 
```
python mig_helper.py --init
```
On the GPU node, go to the service folder depending on which service (e.g., yolo, albert). Run
```
python setup_service.py --system
```
This will generate a ``master.json`` file, which we can send request to its IP address (port 5200) to modify GPU partition/service.
#TODO: make service.json contain all GPU nodes (add node id to key) {nodeid: {pid:{name:xx, port:xx}}


### Generator node

On the generator node, run
```
python generator.py
```
This starts a tcp listener (port 10002), and once it receives `start` signal from controller, it starts a new process of producer and a consumer that processes a certain number of requests, then go back to listening.

# Schemes

## base
Just run max model variant on full GPU. On 2 GPUs, the request arrival interval is 1s, equivalent to 0.2s interval on 10 GPUs.

## random
Randomly explore the vanilla search space of partition and model variant. Uses history to skip evaluation if one partition+variant config is seen in previous hour. Runs on 2 GPUs with 0.2s interval.

## Clover
The technique. Explore in graph space. Use SA algorithm. Runs on 2 GPUs with 0.2s interval.