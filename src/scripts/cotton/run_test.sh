#!/bin/bash
#$ -j 
#$ -o /homes/ykohata/code/devml/homes/ypark/code/seg/trash
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11/3.11.9
source ~/.bashrc
conda activate new

WORKDIR=/homes/ykohata/code/devml/homes/ypark/code/seg

echo "ok"

cd $WORKDIR/src

$seed=101
python test.py voc \
    default.dataset_dir="/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug" \
    default.seed=$seed \
    learn.n_epoch=250 \
    augment.name=["ra"] \
    augment.ra.weight="single" \
    augment.ra.single="ShearX" \
    test.best_model="" \
    result_dir=""