#!/usr/bin/env bash

set -xe

DATA_PATH=$1
DUMP_PATH=$2
TMSPAN=$3
PRE_PATH=model.pt

BERT_CONFIG="--roberta_model drop_dataset/roberta.large"

if [ ${TMSPAN} == tag_mspan ]; then
  echo "Use tag_mspan model..."
  MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
else
  echo "Use mspan model..."
  MODEL_CONFIG="--gcn_steps 3 --use_gcn"
fi

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 1 --pre_path ${PRE_PATH} --data_mode dev --dump_path ${DUMP_PATH} \
             --inf_path ${DATA_PATH}"

python roberta_predict.py \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}
