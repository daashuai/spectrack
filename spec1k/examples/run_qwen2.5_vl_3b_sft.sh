set -x

set -x
ENGINE=${1:-vllm}
HF_ENDPOINT="https://hf-mirror.com"
PROJECT_DIR="/home/ludashuai/spectrack"
export CUDA_VISIBLE_DEVICES=0,1,3,4

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-VL-3B-Instruct \
    trainer.project_name='specktrack-sft' \
    trainer.experiment_name='qwen2_5_vl_3b-sft' \
    trainer.total_epochs=4 \
    trainer.logger='["tensorboard"]'

