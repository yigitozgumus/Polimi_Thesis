#!/bin/bash

# Old model ablation studies for test
python run_on_titan_parameters.py -c configs/32/bigan.json -e bigan_32_abl_sum
python run_on_titan_parameters.py -c configs/32/alad.json -e alad_32_abl_sum
python run_on_titan_parameters.py -c configs/32/ganomaly.json -e ganomaly_32_abl_sum
python run_on_titan_parameters.py -c configs/32/skip_ganomaly.json -e skip_ganomaly_abl_sum
python run_on_titan_parameters.py -c configs/32/anogan.json -e anogan_abl_sum
#python run_on_titan_new.py -c configs/new/sencebgan.json -e sencebgan_exp_res

