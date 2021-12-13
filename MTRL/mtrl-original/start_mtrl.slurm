#!/bin/bash
#SBATCH --job-name=mtrl_care_baseline
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH -c 2

# Source the created env.sh file to get mujoco and python3.6
singularity exec --nv --bind $HOME/mujocopy-buildlock:/ext3/miniconda3/lib/python3.9/site-packages/mujoco_py/generated/mujocopy-buildlock --overlay /scratch/ig2283/tmp/overlay-5GB-200K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate hw2_drq

export PYTHONPATH=$SCRATCH/.local/lib/python3.9/site-packages/

python3.9 -u main.py \
setup=metaworld \
env=metaworld-mt10 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=1 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=10 \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.should_use_task_encoder=True \
agent.encoder.type_to_select=moe \
agent.multitask.should_use_multi_head_policy=False \
agent.encoder.moe.task_id_to_encoder_id_cfg.mode=attention \
agent.encoder.moe.num_experts=4 \
agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=True


"