conda activate DST
python main.py \
--data_rate 1.0 \
--detail_log 1 \
--batch_size 8 \
--test_batch_size 16 \
--save_prefix debugging \
--gpus 1 \
--port 11463 \
--do_train 1 \
--do_short 0 \
--seed 1 \
--max_epoch 3 \
--max_length 256 \
--aux 0
