#!/bin/bash
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:ncpus=24:mem=3G
#PBS -l walltime=05:00:00
#PBS -P Personal
#PBS -o output.txt
#PBS -N contractnli_3jul
# Commands start here
cd ${PBS_O_WORKDIR}
module load python/3.6.0
module load torch/2016-08-02
python train.py ./data/conf_base.yml ./output