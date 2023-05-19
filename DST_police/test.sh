#!/bin/sh

#SBATCH -J  test   # Job name
#SBATCH -o  ./out/test_samsung.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A100       # queue  name  or  partiton name

#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

## 노드 지정하지않기
#SBATCH   --nodes=1

#### Select  GPU
#SBATCH   --gres=gpu:1
#SBTACH   --ntasks=1
#SBATCH   --tasks-per-node=16
#SBATCH     --mail-user=jihyunlee@postech.ac.kr
#SBATCH     --mail-type=ALL

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module purge
module add cuda/11.0
module add cuDNN/cuda/11.0/8.0.4.30
#module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA_new "
conda activate QA_new

export PYTHONPATH=.


TRAIN_DIR=$HOME/t5-woz-kor


python main.py \
--detail_log 1 \
--batch_size 32 \
--test_batch_size 64 \
--pretrained_model './model/wozk0.1.pt' \
--save_prefix k_samsung2 \
--gpus 1 \
--port 11463 \
--do_train 0 \
--do_short 0 \
--seed 1 \
--max_length 256 \
