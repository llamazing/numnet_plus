#!/usr/bin/env bash

set -xe

SEED=$1
LR=$2
BLR=$3
WD=$4
BWD=$5
TMSPAN=$6

BASE_DIR=.

DATA_DIR=${BASE_DIR}/drop_dataset
CODE_DIR=${BASE_DIR}

if [ ${TMSPAN} == tag_mspan ];then
  echo "Use tag_mspan model..."
  CACHED_TRAIN=${DATA_DIR}/tmspan_cached_roberta_train.pkl
  CACHED_DEV=${DATA_DIR}/tmspan_cached_roberta_dev.pkl
  MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${DATA_DIR} --tag_mspan
  fi
else
  echo "Use mspan model..."
  CACHED_TRAIN=${DATA_DIR}/cached_roberta_train.pkl
  CACHED_DEV=${DATA_DIR}/cached_roberta_dev.pkl
  MODEL_CONFIG="--gcn_steps 3 --use_gcn"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --input_path ${DATA_DIR} --output_dir ${DATA_DIR}
  fi
fi


SAVE_DIR=${BASE_DIR}/numnet_plus_${SEED}_LR_${LR}_BLR_${BLR}_WD_${WD}_BWD_${BWD}${TMSPAN}
DATA_CONFIG="--data_dir ${DATA_DIR} --save_dir ${SAVE_DIR}"
TRAIN_CONFIG="--batch_size 16 --eval_batch_size 5 --max_epoch 5 --warmup 0.06 --optimizer adam \
              --learning_rate ${LR} --weight_decay ${WD} --seed ${SEED} --gradient_accumulation_steps 4 \
              --bert_learning_rate ${BLR} --bert_weight_decay ${BWD} --log_per_updates 100 --eps 1e-6"
BERT_CONFIG="--roberta_model ${DATA_DIR}/roberta.large"


echo "Start training..."
python ${CODE_DIR}/roberta_gcn_cli.py \
    ${DATA_CONFIG} \
    ${TRAIN_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 5 --pre_path ${SAVE_DIR}/checkpoint_best.pt --data_mode dev --dump_path ${SAVE_DIR}/dev.json \
             --inf_path ${DATA_DIR}/drop_dataset_dev.json"

python ${CODE_DIR}/roberta_predict.py \
    ${DATA_CONFIG} \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

python ${CODE_DIR}/drop_eval.py \
    --gold_path ${DATA_DIR}/drop_dataset_dev.json \
    --prediction_path ${SAVE_DIR}/dev.json