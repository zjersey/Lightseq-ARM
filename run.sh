#!/bin/bash

export RUN_DIR=/data/local/tmp
cd ${RUN_DIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${RUN_DIR}
export OMP_NUM_THREADS=1
./demo quant_transformer.pb