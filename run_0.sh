#!/bin/bash

python run_klue.py train --task klue-ner --output_dir output2/run_0 --data_dir data/klue-ner --num_train_epochs 10 --max_seq_length 510 --metric_key character_macro_f1 --accelerator gpu --devices 0 --model_name_or_path pretrained-com/KLUE-RoBERTa
python run_klue.py train --task klue-dp --output_dir output2/run_0 --data_dir data/klue-dp --num_train_epochs 20 --max_seq_length 256 --metric_key las_macro_f1 --warmup_ratio 0.1 --accelerator gpu --devices 0 --model_name_or_path pretrained-com/KLUE-RoBERTa
