#!/bin/bash

RUN_NAME=$1

sbatch \
  -J $RUN_NAME \
  -o ${RUN_NAME}%j.out \
  -e ${RUN_NAME}%j.err \
  training.sh $RUN_NAME
