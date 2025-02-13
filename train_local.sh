#!/bin/bash

# docker run --volume "C:\Users\20195435\Documents\theta\projects\cosmo":data chjkusters4/wle-cade-nbi-cadx:V2 /data/Insights-CADe-BE/train_local.sh
# STEP 1: Export folders for further use
export HOME_FOLDER=/data/
export BASE_FOLDER=/data/Insights-CADe-BE
export EXP_FOLDER=/data/experiments

# STEP 2: Navigate to the location of the code
cd $BASE_FOLDER 

# print the current working directory
pwd

# STEP 3: Setup WANDB logging
export WANDB_API_KEY=abe0a54fb25d072906a33bbf9d57a0bd0360ead6
export WANDB_DIR=$BASE_FOLDER/wandb
export WANDB_CONFIG_DIR=$BASE_FOLDER/wandb
export WANDB_CACHE_DIR=$BASE_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login

# STEP 4: Define Models and Seeds
MODELS=('MetaFormer-CAS18-FPN')
SEEDS=(42)

# STEP 5: Execute experiments by for-loops
for i in "${!MODELS[@]}"
do
    for j in "${!SEEDS[@]}"
    do
        export OUTPUT_FOLDER=${MODELS[$i]}_${SEEDS[$j]}
        python3 train.py --experimentname "${OUTPUT_FOLDER}" \
                            --seed "${SEEDS[$j]}" \
                            --output_folder "${OUTPUT_FOLDER}" \
                            --backbone "${EXP_FOLDER}/${MODELS[$i]}" \
                            --weights "${BASE_FOLDER}/pretrained/checkpoint0100_teacher.pth" \
                            --batchsize 8 \
                            --cache_path "/data/barretts_cache"

        echo "============Finished============"
        
    done
done