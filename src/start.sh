#!/bin/bash
master_port=29500
gpu=3
from=0

CURDIR="$(cd `dirname $0`; pwd)"
COUNT=0
args=($@)
for arg in ${args[@]}; do
    if [ "${arg:0:7}" == "--port=" ]
    then
        master_port=${arg:7}
    elif [ "${arg:0:6}" == "--gpu=" ]
    then
        gpu=${arg:6}
    elif [ "${arg:0:7}" == "--from=" ]
    then
        from=${arg:7}
    else
        ARGS[$COUNT]=$arg
        COUNT=`expr $COUNT + 1`
    fi
done
python -m torch.distributed.launch --master_port=$master_port --nproc_per_node=$gpu train.py ${ARGS[@]} --start_from=$from --gpus=$gpu
