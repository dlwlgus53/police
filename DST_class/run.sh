conda activate DST
CUDA_VISIBLE_DEVICES=2,3 python main.py \
--batch_size 16 \
--test_batch_size 32 \
--save_prefix clasification \
--gpus 2 \
--do_train 1 \
--do_short 0 \
--seed 1 \
--max_epoch 20 \
