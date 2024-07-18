#!/bin/bash

# custom config
DATA=path/to/your/data
TRAINER=JoAPR
SHOTS=16
NCTX=16
CSC=False
CTP=end
CFG=rn50

DATASET=$1
FP=$2
FPTYPE=$3

for SEED in 1 2 3
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${DATASET}/${CFG}_${SHOTS}shots_${FP}FP_${FPTYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --model-dir output/${DATASET}/${CFG}_${SHOTS}shots_${FP}FP_${FPTYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 200 \
    --eval-only \
    TRAINER.JOAPR.N_CTX ${NCTX} \
    TRAINER.JOAPR.CSC ${CSC} \
    TRAINER.JOAPR.CLASS_TOKEN_POSITION ${CTP}
done