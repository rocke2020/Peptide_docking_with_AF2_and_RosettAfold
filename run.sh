export CUDA_VISIBLE_DEVICES=2
nohup python Code/Running_and_parsing_jobs/alphafold2_advanced.py \
    > run.log 2>&1 &