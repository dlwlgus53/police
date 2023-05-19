conda activate DST
CUDA_VISIBLE_DEVICES=1 python inference.py \
--base_trained 'kykim/bert-kor-base' \
--pretrained_model '/home/jihyunlee/police/DST_class/model/police_clasification.pt' \
