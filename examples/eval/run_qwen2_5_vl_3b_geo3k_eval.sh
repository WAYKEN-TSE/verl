set -x

data_path=$HOME/verl/data/gen/qwen2_5_vl_3b_rollout_10_gen_test.parquet

python -m verl.trainer.main_eval data.path=$data_path 