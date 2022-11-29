#!/bin/ksh

# La file d'attente gpu : NE PAS CHANGER
#$ -q 'gpu*&!*07"

#$ -o /work/imvia/an3112sc/perso/GeoG/gpu_train.out
#$ -N gpu_train


echo "Starting my shell script"

MYMODULE="pytorch/1.11.0/cuda/11.3.1/gpu"
module load ${MYMODULE}

export PYTHONUSERBASE="/work/imvia/an3112sc/pythonEnvs/perso"
pip install --upgrade torchvision

MYROOTDIR="/work/imvia/an3112sc/perso/GeoG"
cd $MYROOTDIR

echo "================= START PYTHON OUTPUT======================"
python ${MYROOTDIR}/training/train.py
echo "================= END PYTHON OUTPUT======================"
echo "Ending the shell script"