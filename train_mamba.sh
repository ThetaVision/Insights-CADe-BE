#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of A100 Nodes with 4 A100 GPUs (single GPU 128 SBUs/hour, 512 SBUs/hour for an entire node)
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=9                       # Specify the number of CPUs/task (18/GPU, 9/GPU_mig)
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=...                         # Specify the node partition
#SBATCH --time=120:00:00                        # Specify the maximum time the job can run
#SBATCH --mail-type=BEGIN,END                   # Specify when to receive notifications on email
#SBATCH --mail-user=...                         # Specify email address to receive notifications

# STEP 1: Export folders for further use
export HOME_FOLDER=...
export BASE_FOLDER=...
export EXP_FOLDER=experiments

# STEP 2: Navigate to the location of the code
cd $BASE_FOLDER || return

# STEP 3: CREATE .SIF FILE
apptainer pull docker://chjkusters4/wle-cade-nbi-cadx-mamba:V1

# STEP 3: Setup WANDB logging
export WANDB_API_KEY=...
export WANDB_DIR=$BASE_FOLDER/$EXP_FOLDER/wandb
export WANDB_CONFIG_DIR=$BASE_FOLDER/$EXP_FOLDER/wandb
export WANDB_CACHE_DIR=$BASE_FOLDER/$EXP_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login

# STEP 4: Define Models and Seeds
MODELS=('')
SEEDS=(5)

# STEP 5: Execute experiments by for-loops
for i in "${!MODELS[@]}"
do
    for j in "${!SEEDS[@]}"
    do
        export OUTPUT_FOLDER=${MODELS[$i]}_${SEEDS[$j]}
        if [ -f "${HOME_FOLDER}"/$EXP_FOLDER/"${OUTPUT_FOLDER}" ]; then
            echo "Folder for ${OUTPUT_FOLDER} already exists"
            echo "============Skipping============"
        else
            echo "Folder for ${OUTPUT_FOLDER} does not exist"
            echo "============Starting============"
            srun apptainer exec --nv $BASE_FOLDER/wle-cade-nbi-cadx_V2.sif \
            python3 train_cls_mamba.py --experimentname "${OUTPUT_FOLDER}" \
                                       --seed "${SEEDS[$j]}" \
                                       --output_folder ... \
                                       --backbone "${MODELS[$i]}" \
                                       --weights ... \
                                       --optimizer ... \
                                       --scheduler ... \
                                       --cls_criterion ... \
                                       --cls_criterion_weight ... \
                                       --label_smoothing ... \
                                       --batchsize ... \
                                       --augmentations ... \
                                       --mask_content ... \
                                       --training_content ... \
                                       --frame_quality ... \
                                       --frame_perc ... \
                                       --num_epochs ... \
                                       --train_lr ...

            cp -r $BASE_FOLDER/$EXP_FOLDER/"${OUTPUT_FOLDER}" "${HOME_FOLDER}"/$EXP_FOLDER
            echo "============Finished============"
        fi
    done
done