#!/bin/bash

# [mini]
python run_klue.py train --task klue-ner --output_dir output0 --data_dir data/klue-ner-mini --model_name_or_path pretrained-com/KLUE-RoBERTa --num_train_epochs 3 --max_seq_length 64 --metric_key character_macro_f1 --accelerator gpu --devices 2
