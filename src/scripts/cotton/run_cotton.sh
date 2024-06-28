#!/bin/bash
#$ -j 
#$ -o $(dirname $(dirname $(dirname $0)))/trash
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.2
source ~/.bashrc
conda activate new-ra

WORKDIR=$(dirname $(dirname $(dirname $0)))
echo "ok"

cd $WORKDIR/src

seed=101
python main.py