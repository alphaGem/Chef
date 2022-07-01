#torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
#    --save-dir results/gpt-j-int8 \
#    --model gpt-j-full-int8 \
#    --start-lr 1e-4 \
#    --load ../bm-gpt-new.pt

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
    --save-dir results/gpt-j-int8 \
    --start-lr 1e-4 