#!/bin/bash
#python run_on_titan.py -c configs/32/ganomaly.json -e ganomaly_32_sn
#python run_on_titan_new.py -c configs/new/ganomaly_conv.json -e ganomaly_new_sn
python run_on_titan_new.py -c configs/new/fanogan.json -e fanogan_sn_std
python run_on_titan_new.py -c configs/new/ebgan.json -e ebgan_sn
