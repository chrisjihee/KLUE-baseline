#!/bin/bash

# [full]
python run_klue.py train --task klue-dp --output_dir output1 --data_dir data/klue-dp --model_name_or_path pretrained-com/KLUE-RoBERTa --num_train_epochs 10 --max_seq_length 256 --metric_key las_macro_f1 --warmup_ratio 0.1 --gpus 1
