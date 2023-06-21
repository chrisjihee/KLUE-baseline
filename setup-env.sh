#!/bin/bash
conda create -n KLUE-baseline python=3.9 -y
conda activate KLUE-baseline
pip install -r requirements.txt
conda list
