#!/bin/bash

python server_config.py --mode term
sleep 3

MODELS="yolov5x6 yolov5x yolov5l"

for MODEL in $MODELS
do
    for PARTITION in 0 1 3 4 6 17
    do
        echo "Running model $MODEL in partition $PARTITION"
        python server_config.py --mig_config $PARTITION --mode config &&
        sleep 10        
        python server_config.py --mig_config $PARTITION --weights $MODEL --mode start &&
        sleep 30
        echo "Setup done, starting inference"
        python request.py --inf_num 5000 --testcase b8_part_${PARTITION}_${MODEL} &&
        echo "Finished, starting next testcase"
        sleep 3
        python server_config.py --mode term && # end the PIDs of services
        sleep 3
    done        
done    
