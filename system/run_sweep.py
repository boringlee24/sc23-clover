from experiment import BaseExperiment, RandomExperiment
from system.clover import CloverExperiment
import time

models = ['yolo', 'efficientnet', 'albert']
base_variant = ['6', '7', '3']

# first run the base
i = 2 

exp = BaseExperiment(models[i], num_nodes=1, gpus_per_node=2)
ip1, ip2, ip3 = exp.initialize(mig_partition='0', models=base_variant[i])
exp.sweep_arrival('base')
time.sleep(30)

# now run with MIG and variant

partitions = ['1', "1", '1']
variants = ["lx", "33", '01']

exp = BaseExperiment(models[i], num_nodes=1, gpus_per_node=2)
exp.initialize(mig_partition=partitions[i], models=variants[i], wait=False, gpu_node_ip=ip1, generator_ip=ip2, generator_port=ip3)
exp.sweep_arrival('clover')

# exp.run()
# time.sleep(30)

# exp = RandomExperiment('yolo', num_nodes=1, gpus_per_node=2)
# ip1, ip2, ip3 = exp.initialize(mig_partition='0', models='6', wait=False, gpu_node_ip=ip1, generator_ip=ip2, generator_port=ip3)
# exp.run(Lambda=0.1, base_ci=250, base_pue=1.5)

# time.sleep(60)
# print('finished random')

# exp = CloverExperiment('yolo', num_nodes=1, gpus_per_node=2)
# exp.initialize(mig_partition='7', models='xxx', wait=False, gpu_node_ip=ip1, generator_ip=ip2, generator_port=ip3)
# exp.run(Lambda=0.1, base_ci=250, base_pue=1.5)
# print('finished clover')
