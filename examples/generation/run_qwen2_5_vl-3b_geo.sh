set -x

data_path=/root/paddlejob/workspace/env_run/xieweikang/verl/data/geo3k/test.parquet
save_path=$HOME/data/gen/qwen2_5_vl_3b_rollout_10_gen_test.parquet
model_path=/root/paddlejob/workspace/env_run/xieweikang/verl/checkpoints/verl_grpo_example_geo3k/qwen2_5_vl_3b_rollout_10/actor_merged
# model_path=Qwen/Qwen2.5-VL-3B-Instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8  \
    rollout.tensor_model_parallel_size=2
