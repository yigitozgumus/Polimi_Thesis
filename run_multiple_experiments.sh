#!/bin/bash

# Old model ablation studies for test
#python run_on_tesla_parameters.py -c configs/32/bigan.json -e bigan_abl_last
#python run_on_titan_parameters.py -c configs/32/alad.json -e alad_abl_last
python run_on_titan_parameters.py -c configs/32/ganomaly.json -e ganomaly_abl_last
python run_on_titan_parameters.py -c configs/32/skip_ganomaly.json -e skip_ganomaly_abl_last
python run_on_titan_parameters.py -c configs/32/anogan.json -e anogan_abl_last
python run_on_titan_new.py -c configs/new/sencebgan.json -e sencebgan_scaled

