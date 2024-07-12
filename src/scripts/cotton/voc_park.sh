#!/bin/bash
#$ -j 
#$ -o $(dirname $(dirname $(dirname $0)))/trash
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.2
source ~/.bashrc
conda activate new

WORKDIR=$(dirname $(dirname $(dirname $0)))
echo "ok"
echo $WORKDIR

cd $WORKDIR


seed=101
python main.py voc \
    default.dataset_dir="/homes/ypark/code/dataset/" \
    augment.name=["nan"] \
    learn.n_epoch=2
