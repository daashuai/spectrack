set -x
ENGINE=${1:-vllm}
HF_ENDPOINT="https://hf-mirror.com"
PROJECT_DIR="/home/lijiehui/project/spectrack"
export CUDA_VISIBLE_DEVICES=2,3
save_path="/home/lijiehui/project/spectrack/checkpoints/"

torchrun --standalone --nnodes=1 --nproc_per_node=2\
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/lijiehui/project/spectrack/spec1k/dataset/sft/train.parquet \
    data.val_files=/home/lijiehui/project/spectrack/spec1k/dataset/sft/test.parquet \
    data.prompt_key=prompt \
    data.response_key=expert_reason_chain \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=2 \
    +data.prompt_dict_keys=['content']\
    +data.apply_chat_template=false\
    data.truncation='error'\
    +data.max_prompt_length=16384\
    +data.max_response_length=8192\
    data.max_length=8192\
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    trainer.project_name='specktrack-sft' \
    trainer.experiment_name='qwen2_5_3b-sft' \
    trainer.total_epochs=4 \
    trainer.logger='["tensorboard"]' \
    trainer.default_local_dir=$save_path \

