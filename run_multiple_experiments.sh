#!/bin/bash
#python run_on_titan_new.py -c configs/new/ebgan.json -e ebgan_short
python run_on_titan_parameters.py -c configs/32/bigan.json -e bigan_32_m3
python run_on_titan_parameters_new.py -c configs/new/fanogan.json -e fanogan_mark2
#python run_on_titan.py -c configs/32/alad.json -e alad_new_lr
#python run_on_titan.py -c configs/32/ganomaly.json -e ganomaly_new_lr
python run_on_titan.py -c configs/32/skip_ganomaly.json -e skip_ganomaly_new_lr
python run_on_titan_new.py -c configs/new/encebgan.json -e encebgan_test
#python run_on_titan.py -c configs/32/alad.json -e alad_fl_lr
