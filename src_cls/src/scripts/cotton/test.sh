#!/bin/bash
#$ -j 
#$ -o /homes/ykohata/code/devml/homes/ypark/code/seg/trash/
#$ -cwd

source /etc/profile.d/modules.sh
module load  python/3.11/3.11.2
source ~/.bashrc
conda activate new-ra

WORKDIR="/homes/ykohata/code/devml/homes/ypark/code/seg/src_cls/src"
echo "ok"

cd $WORKDIR

seed=112
python main.py voc \
    default.seed=$seed \
    learn.n_epoch=250 \
    augment.name=["ra"] \
    augment.ra.weight="single" \
    augment.ra.single="ShearX" \
    && python notify.py 0 || python notify.py 1