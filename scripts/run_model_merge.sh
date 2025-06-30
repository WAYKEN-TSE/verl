set -x
python scripts/model_merger.py merge \
  --backend=fsdp \
  --local_dir="/root/paddlejob/workspace/env_run/xieweikang/verl/checkpoints/verl_grpo_example_geo3k/qwen2_5_vl_3b_rollout_5/global_step_140/actor" \
  --target_dir="/root/paddlejob/workspace/env_run/xieweikang/verl/checkpoints/verl_grpo_example_geo3k/qwen2_5_vl_3b_rollout_5/actor_merged" \
