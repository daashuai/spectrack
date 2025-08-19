#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export http_proxy="http://127.0.0.1:7898"
export https_proxy="http://127.0.0.1:7898"
export all_proxy="socks5://127.0.0.1:7891"

MODEL_PATH=/data/zch/models/Qwen2.5-3B-Instruct
train_data_path=/data/zch/projects/water_quality_mllm/EasyR1-main/my_code/data/water_mllm_v0/train.parquet
val_data_path=/data/zch/projects/water_quality_mllm/EasyR1-main/my_code/data/water_mllm_v0/val.parquet
format_prompt=/data/zch/projects/water_quality_mllm/EasyR1-main/my_code/prompt_format.jinja
total_epochs=50
project_name=Water_MLLM_v0
experiment_name=v1
save_checkpoint_path=/data/zch/projects/water_quality_mllm/EasyR1-main/my_code/checkpoints_v1

python3 -m verl.trainer.main \
    config=my_code/config.yaml \
    data.train_files=${train_data_path} \
    data.val_files=${val_data_path} \
    data.format_prompt=${format_prompt} \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=/data/zch/projects/water_quality_mllm/EasyR1-main/my_code/custom_reward.py:compute_score \
    trainer.total_epochs=${total_epochs} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger="['console','wandb']" \
    trainer.val_freq=1 \
    trainer.save_freq=1 \
    trainer.save_checkpoint_path=${save_checkpoint_path} \
    trainer.n_gpus_per_node=8
