#!/bin/bash

# custom config
DATA=path/to/your/data
TRAINER=JoAPR

DATASET=$1
FP=$2  # number of noise samples per class (0<FP<=SHOTS)
FPTYPE=$3  #type of noise(symflip, pairflip)
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)


for SEED in 1 2 3
do
    DIR=output/${DATASET}/${CFG}_${SHOTS}shots_${FP}FP_${FPTYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.JOAPR.N_CTX ${NCTX} \
        TRAINER.JOAPR.CSC ${CSC} \
        TRAINER.JOAPR.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_FP ${FP} \
        DATASET.FP_TYPE ${FPTYPE} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done