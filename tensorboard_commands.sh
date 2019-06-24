#!/bin/bash
# add the appropriate tensorboard commands
# BIGAN
if [ $1 == 'bigan' ]
then
    tensorboard --logdir=bigan0:Ablation/bigan_abl_last_000,\
bigan1:Ablation/bigan_abl_last_001,bigan2:Ablation/bigan_abl_last_010,\
bigan3:Ablation/bigan_abl_last_011,bigan4:Ablation/bigan_abl_last_100,\
bigan5:Ablation/bigan_abl_last_101,bigan6:Ablation/bigan_abl_last_110,\
bigan7:Ablation/bigan_abl_last_111
# ALAD
elif [ $1 == 'alad' ]
then
    tensorboard --logdir=alad0:Ablation/alad_abl_last_000,\
alad1:Ablation/alad_abl_last_001,alad2:Ablation/alad_abl_last_010,\
alad3:Ablation/alad_abl_last_011,alad4:Ablation/alad_abl_last_100,\
alad5:Ablation/alad_abl_last_101,alad6:Ablation/alad_abl_last_110,\
alad7:Ablation/alad_abl_last_111
# ANOGAN
elif [ $1 == 'anogan' ]
then
    tensorboard --logdir=anogan0:Ablation/anogan_abl_last_000,\
anogan1:Ablation/anogan_abl_last_001,anogan2:Ablation/anogan_abl_last_010,\
anogan3:Ablation/anogan_abl_last_011,anogan4:Ablation/anogan_abl_last_100,\
anogan5:Ablation/anogan_abl_last_101,anogan6:Ablation/anogan_abl_last_110,\
anogan7:Ablation/anogan_abl_last_111
# GANOMALY
elif [ $1 == 'ganomaly' ]
then
    tensorboard --logdir=ganomaly0:Ablation/ganomaly_abl_last_000,\
ganomaly1:Ablation/ganomaly_abl_last_001,ganomaly2:Ablation/ganomaly_abl_last_010,\
ganomaly3:Ablation/ganomaly_abl_last_011,ganomaly4:Ablation/ganomaly_abl_last_100,\
ganomaly5:Ablation/ganomaly_abl_last_101,ganomaly6:Ablation/ganomaly_abl_last_110,\
ganomaly7:Ablation/ganomaly_abl_last_111
# SKIP_GANOMALY
elif [ $1 == 'skip_ganomaly' ]
then
    tensorboard --logdir=skip_ganomaly0:Ablation/skip_ganomaly_abl_last_000,\
skip_ganomaly1:Ablation/skip_ganomaly_abl_last_001,skip_ganomaly2:Ablation/skip_ganomaly_abl_last_010,\
skip_ganomaly3:Ablation/skip_ganomaly_abl_last_011,skip_ganomaly4:Ablation/skip_ganomaly_abl_last_100,\
skip_ganomaly5:Ablation/skip_ganomaly_abl_last_101,skip_ganomaly6:Ablation/skip_ganomaly_abl_last_110,\
skip_ganomaly7:Ablation/skip_ganomaly_abl_last_111
# SENCEBGAN_1
elif [ $1 == 'sencebgan_1' ]
then
    tensorboard --logdir=sencebgan0:Ablation/senceb_abl_14_False_False,\
sencebgan1:Ablation/senceb_abl_14_False_True,sencebgan2:Ablation/senceb_abl_14_True_False,\
sencebgan3:Ablation/senceb_abl_14_True_True
# SENCEBGAN_2
elif [ $1 == 'sencebgan_2' ]
then
    tensorboard --logdir=sencebgan0:Ablation/sencebgan_abl_last_False_False,\
sencebgan1:Ablation/sencebgan_abl_last_False_true,sencebgan2:Ablation/sencebgan_abl_last_True_False,\
sencebgan3:Ablation/sencebgan_abl_last_True_True

fi

