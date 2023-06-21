#!/bin/bash
conda create -n KLUE-baseline python=3.7 -y
conda activate KLUE-baseline
pip install -r requirements.txt
pip list
