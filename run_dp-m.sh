#!/bin/bash

# [mini]
python run_klue.py train --task klue-dp --output_dir output0 --data_dir data/klue-dp-mini --model_name_or_path pretrained-com/KLUE-RoBERTa --num_train_epochs 3 --max_seq_length 128 --metric_key las_macro_f1 --warmup_ratio 0.1 --accelerator gpu --devices 3
