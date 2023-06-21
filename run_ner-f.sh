#!/bin/bash

# [full]
python run_klue.py train --task klue-ner --output_dir output1 --data_dir data/klue-ner --model_name_or_path pretrained-com/KLUE-RoBERTa --num_train_epochs 3 --max_seq_length 510 --metric_key character_macro_f1 --gpus 0
