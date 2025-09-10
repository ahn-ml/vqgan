#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONWARNINGS="ignore"
python main.py --base configs/vqgan_pusht.yaml \
    --train True \
    --gpus 0,1 \
    --project djepa2_pusht \
    --name L8x8,T96,V512,ch96,res2,batch32x2 \
    --seed 2026