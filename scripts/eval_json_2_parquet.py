import pandas as pd
import numpy as np

pq=pd.read_parquet("/root/paddlejob/workspace/env_run/xieweikang/verl_1/data/geo3k/test.parquet")
js=pd.read_json("/root/paddlejob/workspace/env_run/xieweikang/verl/outputs/val2/60.jsonl",lines=True)

pq['responses']=""
for i in range(len(pq)):
    pq.loc[i,'responses']=[js.loc[i,'output']]
print(pq['responses'])

pq.to_parquet("/root/paddlejob/workspace/env_run/xieweikang/verl_1/data/geo3k/qw_megatron_js_2_pq.parquet",index=False)